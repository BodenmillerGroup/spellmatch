from functools import partial, wraps
from pathlib import Path
from typing import Any, Optional, Type

import click
import click_log
import numpy as np
import pandas as pd
import pluggy
import xarray as xr
import yaml
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)

from . import hookspecs, io
from ._spellmatch import SpellmatchException, logger
from .matching.algorithms import MaskMatchingAlgorithm, icp, probreg, spellmatch
from .registration.feature_based import (
    matcher_types,
    register_image_features,
)
from .registration.intensity_based import (
    register_image_intensities,
    sitk_transform_types,
)
from .registration.intensity_based.sitk_metrics import sitk_metric_types
from .registration.intensity_based.sitk_optimizers import sitk_optimizer_types
from .registration.region_based import register_mask_regions

click_log.basic_config(logger=logger)

transform_types: dict[str, Type[ProjectiveTransform]] = {
    "rigid": EuclideanTransform,
    "similarity": SimilarityTransform,
    "affine": AffineTransform,
}


def get_plugin_manager() -> pluggy.PluginManager:
    pm = pluggy.PluginManager("spellmatch")
    pm.add_hookspecs(hookspecs)
    pm.load_setuptools_entrypoints("spellmatch")
    pm.register(icp, name="spellmatch-icp")
    pm.register(probreg, name="spellmatch-probreg")
    pm.register(spellmatch, name="spellmatch-spellmatch")
    return pm


def catch_exception(func=None, *, handle=SpellmatchException):
    if not func:
        return partial(catch_exception, handle=handle)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except handle as e:
            raise click.ClickException(e)

    return wrapper


def glob_sorted(dir: Path, pattern: str, expect: Optional[int] = None) -> list[Path]:
    files = sorted(dir.glob(pattern))
    if expect is not None and len(files) != expect:
        raise click.UsageError(
            f"Expected {expect} files in directory {dir.name}, found {len(files)}"
        )
    return files


def describe_image(img: xr.DataArray) -> str:
    return f"{np.dtype(img.dtype).name} {img.shape[1:]}, {img.shape[0]} channels"


def describe_mask(mask: xr.DataArray) -> str:
    return f"{np.dtype(mask.dtype).name} {mask.shape}, {len(np.unique(mask)) - 1} cells"


def describe_assignment(assignment: pd.DataFrame) -> str:
    return f"{len(assignment.index)} cell pairs"


def describe_scores(scores: xr.DataArray) -> str:
    top2_scores = -np.partition(-scores, 1, axis=-1)[:, :2]
    mean_score = np.mean(top2_scores[:, 0])
    mean_margin = np.mean(top2_scores[:, 0] - top2_scores[:, 1])
    return f"mean score: {mean_score}, mean margin: {mean_margin}"


def describe_transform(transform: ProjectiveTransform) -> str:
    if type(transform) is ProjectiveTransform:
        transform = AffineTransform(matrix=transform.params)
    transform_infos = []
    if hasattr(transform, "scale"):
        if np.isscalar(transform.scale):
            transform_infos.append(f"scale: {transform.scale:.3f}")
        else:
            transform_infos.append(
                f"scale: sx={transform.scale[0]:.3f} sy={transform.scale[1]:.3f}"
            )
    if hasattr(transform, "rotation"):
        transform_infos.append(f"ccw rotation: {180 * transform.rotation / np.pi:.3f}°")
    if hasattr(transform, "shear"):
        transform_infos.append(f"ccw shear: {180 * transform.shear / np.pi:.3f}°")
    if hasattr(transform, "translation"):
        transform_infos.append(
            "translation: "
            f"tx={transform.translation[0]:.3f} ty={transform.translation[1]:.3f}"
        )
    return ", ".join(transform_infos)


class KeywordArgumentsParamType(click.ParamType):
    name = "keyword arguments"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        if not isinstance(value, str):
            return self.fail(f"{value} is not a string", param=param, ctx=ctx)
        if not value:
            return {}
        key_value_pairs = []
        for key_value_pair_str in value.split(","):
            key_value_pair = key_value_pair_str.split(sep="=", maxsplit=1)
            if len(key_value_pair) != 2:
                return self.fail(
                    f"{key_value_pair_str} is not a valid key-value pair",
                    param=param,
                    ctx=ctx,
                )
            key_value_pairs.append(key_value_pair)
        yaml_doc = "\n".join(f"{key}: {value}" for key, value in key_value_pairs)
        try:
            kwargs = yaml.load(yaml_doc, yaml.Loader)
        except yaml.YAMLError as e:
            return self.fail(f"cannot be parsed as YAML: {e}", param=param, ctx=ctx)
        return kwargs


KEYWORD_ARGUMENTS = KeywordArgumentsParamType()


@click.group(name="spellmatch")
@click.version_option()
@click_log.simple_verbosity_option(logger=logger)
def cli() -> None:
    pass


@cli.group(name="register")
def cli_register() -> None:
    pass


@cli_register.command(name="regions")
@click.argument(
    "source_mask_path",
    metavar="SOURCE_MASK(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "target_mask_path",
    metavar="TARGET_MASK(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-image",
    "--source-images",
    "source_img_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--target-image",
    "--target-images",
    "target_img_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-panel",
    "source_panel_file",
    default="source_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--transform-type",
    "transform_type_name",
    default="rigid",
    show_default=True,
    type=click.Choice(list(transform_types.keys())),
)
@click.argument(
    "assignment_path",
    metavar="ASSIGNMENT(S)",
    type=click.Path(path_type=Path),
)
@click.argument(
    "transform_path",
    metavar="TRANSFORM(S)",
    type=click.Path(path_type=Path),
)
@catch_exception(handle=SpellmatchException)
def cli_register_regions(
    source_mask_path: Path,
    target_mask_path: Path,
    source_img_path: Optional[Path],
    target_img_path: Optional[Path],
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    target_scale: float,
    transform_type_name: str,
    assignment_path: Path,
    transform_path: Path,
) -> None:
    source_panel = None
    if source_panel_file.exists():
        source_panel = io.read_panel(source_panel_file)
    target_panel = None
    if target_panel_file.exists():
        target_panel = io.read_panel(target_panel_file)
    if (
        source_mask_path.is_file()
        and target_mask_path.is_file()
        and (source_img_path is None or source_img_path.is_file())
        and (target_img_path is None or target_img_path.is_file())
        and (not assignment_path.exists() or assignment_path.is_file())
        and (not transform_path.exists() or transform_path.is_file())
    ):
        source_mask_files = [source_mask_path]
        target_mask_files = [target_mask_path]
        if source_img_path is not None:
            source_img_files = [source_img_path]
        else:
            source_img_files = [None]
        if target_img_path is not None:
            target_img_files = [target_img_path]
        else:
            target_img_files = [None]
        assignment_files = [assignment_path]
        transform_files = [transform_path]
    elif (
        source_mask_path.is_dir()
        and target_mask_path.is_dir()
        and (source_img_path is None or source_img_path.is_dir())
        and (target_img_path is None or target_img_path.is_dir())
        and (not assignment_path.exists() or assignment_path.is_dir())
        and (not transform_path.exists() or transform_path.is_dir())
    ):
        source_mask_files = glob_sorted(source_mask_path, "*.tiff")
        target_mask_files = glob_sorted(
            target_mask_path, "*.tiff", expect=len(source_mask_files)
        )
        if source_img_path is not None:
            source_img_files = glob_sorted(
                source_img_path, "*.tiff", expect=len(source_mask_files)
            )
        else:
            source_img_files = [None] * len(source_mask_files)
        if target_img_path is not None:
            target_img_files = glob_sorted(
                target_img_path, "*.tiff", expect=len(target_mask_files)
            )
        else:
            target_img_files = [None] * len(target_mask_files)
        assignment_path.mkdir(exist_ok=True)
        assignment_files = [
            assignment_path / f"assignment{i + 1:03d}.csv"
            for i in range(len(source_mask_files))
        ]
        transform_path.mkdir(exist_ok=True)
        transform_files = [
            transform_path / f"transform{i + 1:03d}.npy"
            for i in range(len(source_mask_files))
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    for i, (
        source_mask_file,
        target_mask_file,
        source_img_file,
        target_img_file,
        assignment_file,
        transform_file,
    ) in enumerate(
        zip(
            source_mask_files,
            target_mask_files,
            source_img_files,
            target_img_files,
            assignment_files,
            transform_files,
        )
    ):
        if len(source_mask_files) > 1:
            logger.info(
                f"########## MASK PAIR {i + 1}/{len(source_mask_files)} ##########"
            )
        source_mask = io.read_mask(source_mask_file, scale=source_scale)
        logger.info(
            f"Source mask: {source_mask_file.name} ({describe_mask(source_mask)})"
        )
        target_mask = io.read_mask(target_mask_file, scale=target_scale)
        logger.info(
            f"Target mask: {target_mask_file.name} ({describe_mask(target_mask)})"
        )
        if source_img_file is not None:
            source_img = io.read_image(
                source_img_file, panel=source_panel, scale=source_scale
            )
            logger.info(
                f"Source image: {source_img_file.name} ({describe_image(source_img)})"
            )
        else:
            source_img = None
            logger.info("Source image: None")
        if target_img_file is not None:
            target_img = io.read_image(
                target_img_file, panel=target_panel, scale=target_scale
            )
            logger.info(
                f"Target image: {target_img_file.name} ({describe_image(target_img)})"
            )
        else:
            target_img = None
            logger.info("Target image: None")
        assignment = None
        if assignment_file.exists():
            assignment = io.read_assignment(assignment_file)
        result = register_mask_regions(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform_type=transform_types[transform_type_name],
            assignment=assignment,
        )
        if result is not None:
            assignment, transform = result
            io.write_assignment(assignment_file, assignment)
            logger.info(
                f"Assignment: {assignment_file.name} "
                f"({describe_assignment(assignment)})"
            )
            if transform is not None:
                io.write_transform(transform_file, transform)
                logger.info(
                    f"Transform: {transform_file.name} "
                    f"({describe_transform(transform)})"
                )
            else:
                logger.info("Transform: None")
        else:
            raise click.Abort()


@cli_register.command("features")
@click.argument(
    "source_img_path",
    metavar="SOURCE_IMAGE(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "target_img_path",
    metavar="TARGET_IMAGE(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-panel",
    "source_panel_file",
    default="source_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--source-channel",
    "source_channel",
    type=click.STRING,
)
@click.option(
    "--target-channel",
    "target_channel",
    type=click.STRING,
)
@click.option(
    "--denoise-source",
    "denoise_source",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--denoise-target",
    "denoise_target",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--blur-source",
    "blur_source",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--blur-target",
    "blur_target",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--orb-args",
    "orb_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
)
@click.option(
    "--matcher",
    "matcher_type_name",
    default="bruteforce",
    show_default=True,
    type=click.Choice(list(matcher_types.keys())),
)
@click.option(
    "--keep",
    "keep_matches_frac",
    default=0.2,
    show_default=True,
    type=click.FloatRange(min=0, max=1, min_open=True),
)
@click.option(
    "--ransac-args",
    "ransac_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
)
@click.option(
    "--show/--no-show",
    "show",
    default=False,
    show_default=True,
)
@click.argument(
    "transform_path",
    metavar="TRANSFORM(S)",
    type=click.Path(path_type=Path),
)
@catch_exception(handle=SpellmatchException)
def cli_register_features(
    source_img_path: Path,
    target_img_path: Path,
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    target_scale: float,
    source_channel: Optional[str],
    target_channel: Optional[str],
    denoise_source: Optional[float],
    denoise_target: Optional[float],
    blur_source: Optional[float],
    blur_target: Optional[float],
    orb_kwargs: dict[str, Any],
    matcher_type_name: str,
    keep_matches_frac: float,
    ransac_kwargs: dict[str, Any],
    show: bool,
    transform_path: Path,
) -> None:
    source_panel = None
    if source_panel_file.exists():
        source_panel = io.read_panel(source_panel_file)
    target_panel = None
    if target_panel_file.exists():
        target_panel = io.read_panel(target_panel_file)
    if (
        source_img_path.is_file()
        and target_img_path.is_file()
        and (not transform_path.exists() or transform_path.is_file())
    ):
        source_img_files = [source_img_path]
        target_img_files = [target_img_path]
        transform_files = [transform_path]
    elif (
        source_img_path.is_dir()
        and target_img_path.is_dir()
        and (not transform_path.exists() or transform_path.is_dir())
    ):
        source_img_files = glob_sorted(source_img_path, "*.tiff")
        target_img_files = glob_sorted(
            target_img_path, "*.tiff", expect=len(source_img_files)
        )
        transform_path.mkdir(exist_ok=True)
        transform_files = [
            transform_path / f"transform{i + 1:03d}.npy"
            for i in range(len(source_img_files))
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    for i, (
        source_img_file,
        target_img_file,
        transform_file,
    ) in enumerate(zip(source_img_files, target_img_files, transform_files)):
        if len(source_img_files) > 1:
            logger.info(
                f"########## IMAGE PAIR {i + 1}/{len(source_img_files)} ##########"
            )
        source_img = io.read_image(
            source_img_file, panel=source_panel, scale=source_scale
        )
        logger.info(
            f"Source image: {source_img_file.name} ({describe_image(source_img)})"
        )
        if source_img.ndim == 3:
            if source_channel is not None:
                if (
                    "c" in source_img.coords
                    and source_channel in source_img.coords["c"]
                ):
                    source_img = source_img.loc[source_channel]
                else:
                    try:
                        source_img = source_img[int(source_channel)]
                    except (ValueError, IndexError):
                        raise click.UsageError(
                            f"Source channel {source_channel} is not a channel name "
                            f"or valid index in source image {source_img_file.name}"
                        )
            else:
                raise click.UsageError(
                    "No channel specified "
                    f"for multi-channel source image {source_img_file.name}"
                )
        target_img = io.read_image(
            target_img_file, panel=target_panel, scale=target_scale
        )
        logger.info(
            f"Target image: {target_img_file.name} ({describe_image(target_img)})"
        )
        if target_img.ndim == 3:
            if target_channel is not None:
                if (
                    "c" in target_img.coords
                    and target_channel in target_img.coords["c"]
                ):
                    target_img = target_img.loc[target_channel]
                else:
                    try:
                        target_img = target_img[int(target_channel)]
                    except (ValueError, IndexError):
                        raise click.UsageError(
                            f"Target channel {target_channel} is not a channel name "
                            f"or valid index in target image {target_img_file.name}"
                        )
            else:
                raise click.UsageError(
                    "No channel specified "
                    f"for multi-channel target image {target_img_file.name}"
                )
        transform = register_image_features(
            source_img,
            target_img,
            orb_kwargs=orb_kwargs,
            matcher_type=matcher_types[matcher_type_name],
            keep_matches_frac=keep_matches_frac,
            ransac_kwargs=ransac_kwargs,
            denoise_source=denoise_source,
            denoise_target=denoise_target,
            blur_source=blur_source,
            blur_target=blur_target,
            show=show,
        )
        io.write_transform(transform_file, transform)
        logger.info(
            f"Transform: {transform_file.name} ({describe_transform(transform)})"
        )


@cli_register.command("intensities")
@click.argument(
    "source_img_path",
    metavar="SOURCE_IMAGE(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "target_img_path",
    metavar="TARGET_IMAGE(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-panel",
    "source_panel_file",
    default="source_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--source-channel",
    "source_channel",
    type=click.STRING,
)
@click.option(
    "--target-channel",
    "target_channel",
    type=click.STRING,
)
@click.option(
    "--denoise-source",
    "denoise_source",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--denoise-target",
    "denoise_target",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--blur-source",
    "blur_source",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--blur-target",
    "blur_target",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--metric",
    "sitk_metric_type_name",
    default="correlation",
    show_default=True,
    type=click.Choice(list(sitk_metric_types.keys())),
)
@click.option(
    "--metric-args",
    "sitk_metric_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
)
@click.option(
    "--optimizer",
    "sitk_optimizer_type_name",
    default="regular_step_gradient_descent",
    show_default=True,
    type=click.Choice(list(sitk_optimizer_types.keys())),
)
@click.option(
    "--optimizer-args",
    "sitk_optimizer_kwargs",
    default=",".join(
        [
            "lr=2.0",
            "min_step=1e-4",
            "num_iter=500",
            "grad_magnitude_tol=1e-8",
            "scales=from_index_shift",
        ]
    ),
    show_default=True,
    type=KEYWORD_ARGUMENTS,
)
@click.option(
    "--transform-type",
    "sitk_transform_type_name",
    default="rigid",
    show_default=True,
    type=click.Choice(list(sitk_transform_types.keys())),
)
@click.option(
    "--initial-transform",
    "--initial-transforms",
    "initial_transform_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--show/--no-show",
    "show",
    default=False,
    show_default=True,
)
@click.option(
    "--hold/--no-hold",
    "hold",
    default=False,
    show_default=True,
)
@click.argument(
    "transform_path",
    metavar="TRANSFORM(S)",
    type=click.Path(path_type=Path),
)
@catch_exception(handle=SpellmatchException)
def cli_register_intensities(
    source_img_path: Path,
    target_img_path: Path,
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    target_scale: float,
    source_channel: Optional[str],
    target_channel: Optional[str],
    denoise_source: Optional[float],
    denoise_target: Optional[float],
    blur_source: Optional[float],
    blur_target: Optional[float],
    sitk_metric_type_name: str,
    sitk_metric_kwargs: dict[str, Any],
    sitk_optimizer_type_name: str,
    sitk_optimizer_kwargs: dict[str, Any],
    sitk_transform_type_name: str,
    initial_transform_path: Optional[Path],
    show: bool,
    hold: bool,
    transform_path: Path,
) -> None:
    sitk_metric = sitk_metric_types[sitk_metric_type_name](**sitk_metric_kwargs)
    sitk_optimizer = sitk_optimizer_types[sitk_optimizer_type_name](
        **sitk_optimizer_kwargs
    )
    source_panel = None
    if source_panel_file.exists():
        source_panel = io.read_panel(source_panel_file)
    target_panel = None
    if target_panel_file.exists():
        target_panel = io.read_panel(target_panel_file)
    if (
        source_img_path.is_file()
        and target_img_path.is_file()
        and (initial_transform_path is None or initial_transform_path.is_file())
        and (not transform_path.exists() or transform_path.is_file())
    ):
        source_img_files = [source_img_path]
        target_img_files = [target_img_path]
        initial_transform_files = [initial_transform_path]
        transform_files = [transform_path]
    elif (
        source_img_path.is_dir()
        and target_img_path.is_dir()
        and (initial_transform_path is None or initial_transform_path.is_dir())
        and (not transform_path.exists() or transform_path.is_dir())
    ):
        source_img_files = glob_sorted(source_img_path, "*.tiff")
        target_img_files = glob_sorted(
            target_img_path, "*.tiff", expect=len(source_img_files)
        )
        if initial_transform_path is not None:
            initial_transform_files = glob_sorted(
                initial_transform_path, "*.npy", expect=len(source_img_files)
            )
        else:
            initial_transform_files = [None] * len(source_img_files)
        transform_path.mkdir(exist_ok=True)
        transform_files = [
            transform_path / f"transform{i + 1:03d}.npy"
            for i in range(len(source_img_files))
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    for i, (
        source_img_file,
        target_img_file,
        initial_transform_file,
        transform_file,
    ) in enumerate(
        zip(
            source_img_files, target_img_files, initial_transform_files, transform_files
        )
    ):
        if len(source_img_files) > 1:
            logger.info(
                f"########## IMAGE PAIR {i + 1}/{len(source_img_files)} ##########"
            )
        source_img = io.read_image(
            source_img_file, panel=source_panel, scale=source_scale
        )
        logger.info(
            f"Source image: {source_img_file.name} ({describe_image(source_img)})"
        )
        if source_img.ndim == 3:
            if source_channel is not None:
                if (
                    "c" in source_img.coords
                    and source_channel in source_img.coords["c"]
                ):
                    source_img = source_img.loc[source_channel]
                else:
                    try:
                        source_img = source_img[int(source_channel)]
                    except (ValueError, IndexError):
                        raise click.UsageError(
                            f"Source channel {source_channel} is not a channel name "
                            f"or valid index in source image {source_img_file.name}"
                        )
            else:
                raise click.UsageError(
                    "No channel specified "
                    f"for multi-channel source image {source_img_file.name}"
                )
        target_img = io.read_image(
            target_img_file, panel=target_panel, scale=target_scale
        )
        logger.info(
            f"Target image: {target_img_file.name} ({describe_image(target_img)})"
        )
        if target_img.ndim == 3:
            if target_channel is not None:
                if (
                    "c" in target_img.coords
                    and target_channel in target_img.coords["c"]
                ):
                    target_img = target_img.loc[target_channel]
                else:
                    try:
                        target_img = target_img[int(target_channel)]
                    except (ValueError, IndexError):
                        raise click.UsageError(
                            f"Target channel {target_channel} is not a channel name "
                            f"or valid index in target image {target_img_file.name}"
                        )
            else:
                raise click.UsageError(
                    "No channel specified "
                    f"for multi-channel target image {target_img_file.name}"
                )
        if initial_transform_file is not None:
            initial_transform = io.read_transform(initial_transform_file)
            logger.info(
                f"Initial transform: {initial_transform_file.name} "
                f"({describe_transform(initial_transform)})"
            )
        else:
            initial_transform = None
            logger.info("Initial transform: None")
        transform = register_image_intensities(
            source_img,
            target_img,
            sitk_metric,
            sitk_optimizer,
            sitk_transform_type=sitk_transform_types[sitk_transform_type_name],
            initial_transform=initial_transform,
            denoise_source=denoise_source,
            denoise_target=denoise_target,
            blur_source=blur_source,
            blur_target=blur_target,
            show=show,
            hold=hold,
        )
        io.write_transform(transform_file, transform)
        logger.info(
            f"Transform: {transform_file.name} ({describe_transform(transform)})"
        )


@cli.command(name="match")
@click.argument(
    "source_mask_path",
    metavar="SOURCE_MASK(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "target_mask_path",
    metavar="TARGET_MASK(S)",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-image",
    "--source-images",
    "source_img_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--target-image",
    "--target-images",
    "target_img_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-panel",
    "source_panel_file",
    default="source_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--algorithm",
    "mask_matching_algorithm_name",
    default="icp",
    show_default=True,
    type=click.STRING,
)
@click.option(
    "--algorithm-args",
    "mask_matching_algorithm_kwargs",
    default="max_iter=10,top_k_estim=50,max_dist=5.0",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
)
@click.option(
    "--transform",
    "--transforms",
    "transform_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--reverse/--no-reverse",
    "reverse",
    default=False,
    show_default=True,
)
@click.argument(
    "scores_path",
    metavar="SCORES",
    type=click.Path(path_type=Path),
)
@catch_exception(handle=SpellmatchException)
def cli_match(
    source_mask_path: Path,
    target_mask_path: Path,
    source_img_path: Optional[Path],
    target_img_path: Optional[Path],
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    target_scale: float,
    mask_matching_algorithm_name: str,
    mask_matching_algorithm_kwargs: dict[str, Any],
    transform_path: Optional[Path],
    reverse: bool,
    scores_path: Path,
) -> None:
    pm = get_plugin_manager()
    mask_matching_algorithm_type: Type[
        MaskMatchingAlgorithm
    ] = pm.hook.spellmatch_get_mask_matching_algorithm(
        name=mask_matching_algorithm_name
    )
    mask_matching_algorithm = mask_matching_algorithm_type(
        **mask_matching_algorithm_kwargs
    )
    source_panel = None
    if source_panel_file.exists():
        source_panel = io.read_panel(source_panel_file)
    target_panel = None
    if target_panel_file.exists():
        target_panel = io.read_panel(target_panel_file)
    if (
        source_mask_path.is_file()
        and target_mask_path.is_file()
        and (source_img_path is None or source_img_path.is_file())
        and (target_img_path is None or target_img_path.is_file())
        and (transform_path is None or transform_path.is_file())
        and (not scores_path.exists() or scores_path.is_file())
    ):
        source_mask_files = [source_mask_path]
        target_mask_files = [target_mask_path]
        source_img_files = [source_img_path]
        target_img_files = [target_img_path]
        transform_files = [transform_path]
        scores_files = [scores_path]
    elif (
        source_mask_path.is_dir()
        and target_mask_path.is_dir()
        and (source_img_path is None or source_img_path.is_dir())
        and (target_img_path is None or target_img_path.is_dir())
        and (transform_path is None or transform_path.is_dir())
        and (not scores_path.exists() or scores_path.is_dir())
    ):
        source_mask_files = glob_sorted(source_mask_path, "*.tiff")
        target_mask_files = glob_sorted(
            target_mask_path, "*.tiff", expect=len(source_mask_files)
        )
        if source_img_path is not None:
            source_img_files = glob_sorted(
                source_img_path, "*.tiff", expect=len(source_mask_files)
            )
        else:
            source_img_files = [None] * len(source_mask_files)
        if target_img_path is not None:
            target_img_files = glob_sorted(
                target_img_path, "*.tiff", expect=len(target_mask_files)
            )
        else:
            target_img_files = [None] * len(target_mask_files)
        if transform_path is not None:
            transform_files = glob_sorted(
                transform_path, "*.npy", expect=len(source_mask_files)
            )
        else:
            transform_files = [None] * len(source_mask_files)
        scores_path.mkdir(exist_ok=True)
        scores_files = [
            scores_path / f"scores{i + 1:03d}.npy"
            for i in range(len(source_mask_files))
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    for (
        source_mask_file,
        target_mask_file,
        source_img_file,
        target_img_file,
        transform_file,
        scores_file,
    ) in zip(
        source_mask_files,
        target_mask_files,
        source_img_files,
        target_img_files,
        transform_files,
        scores_files,
    ):
        source_mask = io.read_mask(source_mask_file, scale=source_scale)
        logger.info(
            f"Source mask: {source_mask_file.name} ({describe_mask(source_mask)})"
        )
        target_mask = io.read_mask(target_mask_file, scale=target_scale)
        logger.info(
            f"Target mask: {target_mask_file.name} ({describe_mask(target_mask)})"
        )
        if source_img_file is not None:
            source_img = io.read_image(
                source_img_file, panel=source_panel, scale=source_scale
            )
            logger.info(
                f"Source image: {source_img_file.name} ({describe_image(source_img)})"
            )
        else:
            source_img = None
            logger.info("Source image: None")
        if target_img_file is not None:
            target_img = io.read_image(
                target_img_file, panel=target_panel, scale=target_scale
            )
            logger.info(
                f"Target image: {target_img_file.name} ({describe_image(target_img)})"
            )
        else:
            target_img = None
            logger.info("Target image: None")
        if transform_file is not None:
            transform = io.read_transform(transform_file)
            logger.info(
                f"Transform: {transform_file.name} ({describe_transform(transform)})"
            )
        else:
            transform = None
            logger.info("Transform: None")
        if reverse:
            source_mask, target_mask = target_mask, source_mask
            source_img, target_img = target_img, source_img
            if transform is not None:
                transform = np.linalg.inv(transform)
        scores = mask_matching_algorithm.match_masks(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )
        io.write_scores(scores_file, scores)
        logger.info(f"Scores: {scores_file.name} ({describe_scores(scores)})")


if __name__ == "__main__":
    cli()
