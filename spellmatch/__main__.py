import logging
from collections.abc import Mapping
from functools import partial, wraps
from pathlib import Path
from typing import Any, Optional, OrderedDict

import click
import click_log
import numpy as np
import yaml
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)

from . import io
from ._spellmatch import SpellmatchException, logger
from .assignment import (
    AssignmentDirection,
    assign,
    show_assignment,
    validate_assignment,
)
from .matching.algorithms import MaskMatchingAlgorithm
from .plugins import plugin_manager
from .registration.feature_based import (
    cv2_feature_types,
    cv2_matcher_types,
    register_image_features,
)
from .registration.intensity_based import (
    register_image_intensities,
    sitk_transform_types,
)
from .registration.intensity_based.sitk_metrics import sitk_metric_types
from .registration.intensity_based.sitk_optimizers import sitk_optimizer_types
from .registration.interactive import register_interactive
from .utils import (
    describe_assignment,
    describe_image,
    describe_mask,
    describe_scores,
    describe_transform,
    preprocess_image,
)

# click_log.basic_config(logger=logger)
logger_handler = click_log.ClickHandler()
logger_handler.formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger.handlers = [logger_handler]
logger.propagate = False

mask_matching_algorithm_names = [
    mask_matching_algorithm_name
    for sublist in plugin_manager.hook.spellmatch_get_mask_matching_algorithm(name=None)
    for mask_matching_algorithm_name in sublist
]

transform_types: dict[str, type[ProjectiveTransform]] = {
    "rigid": EuclideanTransform,
    "similarity": SimilarityTransform,
    "affine": AffineTransform,
}


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


class OrderedGroup(click.Group):
    def __init__(
        self,
        name: Optional[str] = None,
        commands: Optional[Mapping[str, click.Command]] = None,
        **kwargs,
    ):
        super(OrderedGroup, self).__init__(name, commands, **kwargs)
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


class KeywordArgumentsParamType(click.ParamType):
    name = "Keyword arguments"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        if not isinstance(value, str):
            self.fail(f"{value} is not a string", param=param, ctx=ctx)
        if not value:
            return {}
        key_val_pairs = []
        for key_val_pair_str in value.split(sep=","):
            key_val_pair = key_val_pair_str.strip().split(sep="=", maxsplit=1)
            if len(key_val_pair) != 2:
                self.fail(
                    f"'{key_val_pair_str}' is not a valid key-value pair",
                    param=param,
                    ctx=ctx,
                )
            key_val_pairs.append((key_val_pair[0].strip(), key_val_pair[1].strip()))
        yaml_doc = "\n".join(f"{key}: {value}" for key, value in key_val_pairs)
        try:
            kwargs = yaml.load(yaml_doc, yaml.Loader)
        except yaml.YAMLError as e:
            self.fail(f"'{value}' cannot be parsed as YAML: {e}", param=param, ctx=ctx)
        return kwargs


KEYWORD_ARGUMENTS = KeywordArgumentsParamType()


@click.group(name="spellmatch", cls=OrderedGroup)
@click.version_option()
def cli() -> None:
    pass


@cli.group(
    name="register",
    cls=OrderedGroup,
    help="Align images (obtain spatial alignment prior)",
)
def cli_register() -> None:
    pass


@cli_register.command(name="interactive", help="Interactive cell matching")
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
    help="Source image file(s)",
)
@click.option(
    "--target-image",
    "--target-images",
    "target_img_path",
    type=click.Path(exists=True, path_type=Path),
    help="Target image file(s)",
)
@click.option(
    "--source-panel",
    "source_panel_file",
    default="source_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Source panel file",
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Target panel file",
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (all axes)",
)
@click.option(
    "--source-zscale",
    "source_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (z-axis)",
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (all axes)",
)
@click.option(
    "--target-zscale",
    "target_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (z-axis)",
)
@click.option(
    "--transform-type",
    "transform_type_name",
    default="rigid",
    show_default=True,
    type=click.Choice(list(transform_types.keys())),
    help="Transformation model",
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
@click_log.simple_verbosity_option(logger=logger)
@catch_exception(handle=SpellmatchException)
def cli_register_interactive(
    source_mask_path: Path,
    target_mask_path: Path,
    source_img_path: Optional[Path],
    target_img_path: Optional[Path],
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    source_zscale: Optional[float],
    target_scale: float,
    target_zscale: Optional[float],
    transform_type_name: str,
    assignment_path: Path,
    transform_path: Path,
) -> None:
    transform_type = transform_types[transform_type_name]
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
            assignment_path
            / f"assignment_{source_mask_file.stem}_to_{target_mask_file.stem}.csv"
            for source_mask_file, target_mask_file in zip(
                source_mask_files, target_mask_files
            )
        ]
        transform_path.mkdir(exist_ok=True)
        transform_files = [
            transform_path
            / f"transform_{source_mask_file.stem}_to_{target_mask_file.stem}.npy"
            for source_mask_file, target_mask_file in zip(
                source_mask_files, target_mask_files
            )
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
            logger.info(f"MASK PAIR {i + 1}/{len(source_mask_files)}")
        source_mask = io.read_mask(
            source_mask_file, scale=source_scale, zscale=source_zscale
        )
        logger.info(
            f"Source mask: {source_mask_file.name} ({describe_mask(source_mask)})"
        )
        target_mask = io.read_mask(
            target_mask_file, scale=target_scale, zscale=target_zscale
        )
        logger.info(
            f"Target mask: {target_mask_file.name} ({describe_mask(target_mask)})"
        )
        if source_img_file is not None:
            source_img = io.read_image(
                source_img_file,
                panel=source_panel,
                scale=source_scale,
                zscale=source_zscale,
            )
            logger.info(
                f"Source image: {source_img_file.name} ({describe_image(source_img)})"
            )
        else:
            source_img = None
            logger.info("Source image: None")
        if target_img_file is not None:
            target_img = io.read_image(
                target_img_file,
                panel=target_panel,
                scale=target_scale,
                zscale=target_zscale,
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
        result = register_interactive(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform_type=transform_type,
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


@cli_register.command("features", help="Feature-based image registration")
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
    help="Source panel",
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Target panel",
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (all axes)",
)
@click.option(
    "--source-zscale",
    "source_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (z-axis)",
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (all axes)",
)
@click.option(
    "--target-zscale",
    "target_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (z-axis)",
)
@click.option(
    "--source-channel",
    "source_channel",
    type=click.STRING,
    help="Source channel name",
)
@click.option(
    "--target-channel",
    "target_channel",
    type=click.STRING,
    help="Target channel name",
)
@click.option(
    "--denoise-source",
    "source_median_filter_size",
    type=click.IntRange(min=3),
    help="Source median filter size",
)
@click.option(
    "--denoise-target",
    "target_median_filter_size",
    type=click.IntRange(min=3),
    help="Target median filter size",
)
@click.option(
    "--clip-source",
    "source_clipping_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Source clipping quantile",
)
@click.option(
    "--clip-target",
    "target_clipping_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Target clipping quantile",
)
@click.option(
    "--blur-source",
    "source_gaussian_filter_sigma",
    type=click.FloatRange(min=0, min_open=True),
    help="Source Gaussian filter SD",
)
@click.option(
    "--blur-target",
    "target_gaussian_filter_sigma",
    type=click.FloatRange(min=0, min_open=True),
    help="Target Gaussian filter SD",
)
@click.option(
    "--feature",
    "cv2_feature_type_name",
    default="ORB",
    show_default=True,
    type=click.Choice(list(cv2_feature_types.keys())),
    help="OpenCV feature",
)
@click.option(
    "--feature-args",
    "cv2_feature_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
    help="OpenCV feature options",
)
@click.option(
    "--matcher",
    "cv2_matcher_type_name",
    default="bruteforce",
    show_default=True,
    type=click.Choice(list(cv2_matcher_types.keys())),
    help="OpenCV matcher",
)
@click.option(
    "--keep",
    "keep_matches_frac",
    default=0.2,
    show_default=True,
    type=click.FloatRange(min=0, max=1, min_open=True),
    help="Fraction of matches to keep",
)
@click.option(
    "--ransac-args",
    "ransac_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
    help="RANSAC options",
)
@click.option(
    "--show/--no-show",
    "show",
    default=False,
    show_default=True,
    help="Enable visualization",
)
@click.argument(
    "transform_path",
    metavar="TRANSFORM(S)",
    type=click.Path(path_type=Path),
)
@click_log.simple_verbosity_option(logger=logger)
@catch_exception(handle=SpellmatchException)
def cli_register_features(
    source_img_path: Path,
    target_img_path: Path,
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    source_zscale: Optional[float],
    target_scale: float,
    target_zscale: Optional[float],
    source_channel: Optional[str],
    target_channel: Optional[str],
    source_median_filter_size: Optional[int],
    target_median_filter_size: Optional[int],
    source_clipping_quantile: Optional[float],
    target_clipping_quantile: Optional[float],
    source_gaussian_filter_sigma: Optional[float],
    target_gaussian_filter_sigma: Optional[float],
    cv2_feature_type_name: str,
    cv2_feature_kwargs: Mapping[str, Any],
    cv2_matcher_type_name: str,
    keep_matches_frac: float,
    ransac_kwargs: Mapping[str, Any],
    show: bool,
    transform_path: Path,
) -> None:
    cv2_feature_type = cv2_feature_types[cv2_feature_type_name]
    cv2_matcher_type = cv2_matcher_types[cv2_matcher_type_name]
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
            transform_path
            / f"transform_{source_img_file.stem}_to_{target_img_file.stem}.npy"
            for source_img_file, target_img_file in zip(
                source_img_files, target_img_files
            )
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
            logger.info(f"IMAGE PAIR {i + 1}/{len(source_img_files)}")
        source_img = io.read_image(
            source_img_file,
            panel=source_panel,
            scale=source_scale,
            zscale=source_zscale,
        )
        logger.info(
            f"Source image: {source_img_file.name} ({describe_image(source_img)})"
        )
        if source_img.ndim in (3, 4):
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
        preprocess_image(
            source_img,
            median_filter_size=source_median_filter_size,
            clipping_quantile=source_clipping_quantile,
            gaussian_filter_sigma=source_gaussian_filter_sigma,
            inplace=True,
        )
        target_img = io.read_image(
            target_img_file,
            panel=target_panel,
            scale=target_scale,
            zscale=target_zscale,
        )
        logger.info(
            f"Target image: {target_img_file.name} ({describe_image(target_img)})"
        )
        if target_img.ndim in (3, 4):
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
        preprocess_image(
            target_img,
            median_filter_size=target_median_filter_size,
            clipping_quantile=target_clipping_quantile,
            gaussian_filter_sigma=target_gaussian_filter_sigma,
            inplace=True,
        )
        transform = register_image_features(
            source_img,
            target_img,
            cv2_feature_type(**cv2_feature_kwargs),
            cv2_matcher_type=cv2_matcher_type,
            keep_matches_frac=keep_matches_frac,
            ransac_kwargs=ransac_kwargs,
            show=show,
        )
        if transform is not None:
            io.write_transform(transform_file, transform)
            logger.info(
                f"Transform: {transform_file.name} ({describe_transform(transform)})"
            )
        else:
            logger.error(
                f"Failed to register {source_img_file.name} and {target_img_file.name}"
            )


@cli_register.command("intensities", help="Intensity-based image registration")
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
    help="Source panel file",
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Target panel file",
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (all axes)",
)
@click.option(
    "--source-zscale",
    "source_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (z-axis)",
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (all axes)",
)
@click.option(
    "--target-zscale",
    "target_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (z-axis)",
)
@click.option(
    "--source-channel",
    "source_channel",
    type=click.STRING,
    help="Source channel name",
)
@click.option(
    "--target-channel",
    "target_channel",
    type=click.STRING,
    help="Target channel name",
)
@click.option(
    "--denoise-source",
    "source_median_filter_size",
    type=click.IntRange(min=3),
    help="Source median filter size",
)
@click.option(
    "--denoise-target",
    "target_median_filter_size",
    type=click.IntRange(min=3),
    help="Target median filter size",
)
@click.option(
    "--clip-source",
    "source_clipping_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Source clipping quantile",
)
@click.option(
    "--clip-target",
    "target_clipping_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Target clipping quantile",
)
@click.option(
    "--blur-source",
    "source_gaussian_filter_sigma",
    type=click.FloatRange(min=0, min_open=True),
    help="Source Gaussian filter SD",
)
@click.option(
    "--blur-target",
    "target_gaussian_filter_sigma",
    type=click.FloatRange(min=0, min_open=True),
    help="Target Gaussian filter SD",
)
@click.option(
    "--metric",
    "sitk_metric_type_name",
    default="correlation",
    show_default=True,
    type=click.Choice(list(sitk_metric_types.keys())),
    help="SimpleITK metric",
)
@click.option(
    "--metric-args",
    "sitk_metric_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
    help="SimpleITK metric options",
)
@click.option(
    "--optimizer",
    "sitk_optimizer_type_name",
    default="regular_step_gradient_descent",
    show_default=True,
    type=click.Choice(list(sitk_optimizer_types.keys())),
    help="SimpleITK optimizer",
)
@click.option(
    "--optimizer-args",
    "sitk_optimizer_kwargs",
    default=",".join(
        [
            "lr=2.0",
            "min_step=1.0e-4",
            "num_iter=500",
            "grad_magnitude_tol=1.0e-8",
            "scales=from_index_shift",
        ]
    ),
    show_default=True,
    type=KEYWORD_ARGUMENTS,
    help="SimpleITK optimizer options",
)
@click.option(
    "--transform-type",
    "sitk_transform_type_name",
    default="rigid",
    show_default=True,
    type=click.Choice(list(sitk_transform_types.keys())),
    help="Transformation model",
)
@click.option(
    "--initial-transform",
    "--initial-transforms",
    "initial_transform_path",
    type=click.Path(exists=True, path_type=Path),
    help="Initial transformation file",
)
@click.option(
    "--show/--no-show",
    "show",
    default=False,
    show_default=True,
    help="Enable visualization",
)
@click.option(
    "--hold/--no-hold",
    "hold",
    default=False,
    show_default=True,
    help="Pause on visualization",
)
@click.argument(
    "transform_path",
    metavar="TRANSFORM(S)",
    type=click.Path(path_type=Path),
)
@click_log.simple_verbosity_option(logger=logger)
@catch_exception(handle=SpellmatchException)
def cli_register_intensities(
    source_img_path: Path,
    target_img_path: Path,
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    source_zscale: Optional[float],
    target_scale: float,
    target_zscale: Optional[float],
    source_channel: Optional[str],
    target_channel: Optional[str],
    source_median_filter_size: Optional[int],
    target_median_filter_size: Optional[int],
    source_clipping_quantile: Optional[float],
    target_clipping_quantile: Optional[float],
    source_gaussian_filter_sigma: Optional[float],
    target_gaussian_filter_sigma: Optional[float],
    sitk_metric_type_name: str,
    sitk_metric_kwargs: Mapping[str, Any],
    sitk_optimizer_type_name: str,
    sitk_optimizer_kwargs: Mapping[str, Any],
    sitk_transform_type_name: str,
    initial_transform_path: Optional[Path],
    show: bool,
    hold: bool,
    transform_path: Path,
) -> None:
    sitk_metric_type = sitk_metric_types[sitk_metric_type_name]
    sitk_optimizer_type = sitk_optimizer_types[sitk_optimizer_type_name]
    sitk_transform_type = sitk_transform_types[sitk_transform_type_name]
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
            transform_path
            / f"transform_{source_img_file.stem}_to_{target_img_file.stem}.npy"
            for source_img_file, target_img_file in zip(
                source_img_files, target_img_files
            )
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
            logger.info(f"IMAGE PAIR {i + 1}/{len(source_img_files)}")
        source_img = io.read_image(
            source_img_file,
            panel=source_panel,
            scale=source_scale,
            zscale=source_zscale,
        )
        logger.info(
            f"Source image: {source_img_file.name} ({describe_image(source_img)})"
        )
        if source_img.ndim in (3, 4):
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
        preprocess_image(
            source_img,
            median_filter_size=source_median_filter_size,
            clipping_quantile=source_clipping_quantile,
            gaussian_filter_sigma=source_gaussian_filter_sigma,
            inplace=True,
        )
        target_img = io.read_image(
            target_img_file,
            panel=target_panel,
            scale=target_scale,
            zscale=target_zscale,
        )
        logger.info(
            f"Target image: {target_img_file.name} ({describe_image(target_img)})"
        )
        if target_img.ndim in (3, 4):
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
        preprocess_image(
            target_img,
            median_filter_size=target_median_filter_size,
            clipping_quantile=target_clipping_quantile,
            gaussian_filter_sigma=target_gaussian_filter_sigma,
            inplace=True,
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
            sitk_metric_type(**sitk_metric_kwargs),
            sitk_optimizer_type(**sitk_optimizer_kwargs),
            sitk_transform_type=sitk_transform_type,
            initial_transform=initial_transform,
            show=show,
            hold=hold,
        )
        io.write_transform(transform_file, transform)
        logger.info(
            f"Transform: {transform_file.name} ({describe_transform(transform)})"
        )


@cli.command(
    name="match", help="Match cells (generate continuous cell alignment scores)"
)
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
    "--algorithm",
    "mask_matching_algorithm_name",
    required=True,
    type=click.Choice(mask_matching_algorithm_names),
    help="Matching algorithm",
)
@click.option(
    "--algorithm-args",
    "mask_matching_algorithm_kwargs",
    default="",
    show_default=True,
    type=KEYWORD_ARGUMENTS,
    help="Matching algorithm options",
)
@click.option(
    "--source-image",
    "--source-images",
    "source_img_path",
    type=click.Path(exists=True, path_type=Path),
    help="Source image file(s)",
)
@click.option(
    "--target-image",
    "--target-images",
    "target_img_path",
    type=click.Path(exists=True, path_type=Path),
    help="Target image file(s)",
)
@click.option(
    "--source-panel",
    "source_panel_file",
    default="source_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Source panel file",
)
@click.option(
    "--target-panel",
    "target_panel_file",
    default="target_panel.csv",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Target panel file",
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (all axes)",
)
@click.option(
    "--source-zscale",
    "source_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (z-axis)",
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (all axes)",
)
@click.option(
    "--target-zscale",
    "target_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (z-axis)",
)
@click.option(
    "--denoise-source",
    "source_median_filter_size",
    type=click.IntRange(min=3),
    help="Source median filter size",
)
@click.option(
    "--denoise-target",
    "target_median_filter_size",
    type=click.IntRange(min=3),
    help="Target median filter size",
)
@click.option(
    "--clip-source",
    "source_clipping_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Source clipping quantile",
)
@click.option(
    "--clip-target",
    "target_clipping_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Target clipping quantile",
)
@click.option(
    "--blur-source",
    "source_gaussian_filter_sigma",
    type=click.FloatRange(min=0, min_open=True),
    help="Source Gaussian filter SD",
)
@click.option(
    "--blur-target",
    "target_gaussian_filter_sigma",
    type=click.FloatRange(min=0, min_open=True),
    help="Target Gaussian filter SD",
)
@click.option(
    "--prior-transform",
    "--prior-transforms",
    "prior_transform_path",
    type=click.Path(exists=True, path_type=Path),
    help="Spatial alignment prior file",
)
@click.option(
    "--reverse/--no-reverse",
    "reverse",
    default=False,
    show_default=True,
    help="Enable target-to-source matching",
)
@click.argument(
    "scores_path",
    metavar="SCORES",
    type=click.Path(path_type=Path),
)
@click_log.simple_verbosity_option(logger=logger)
@catch_exception(handle=SpellmatchException)
def cli_match(
    source_mask_path: Path,
    target_mask_path: Path,
    mask_matching_algorithm_name: str,
    mask_matching_algorithm_kwargs: Mapping[str, Any],
    source_img_path: Optional[Path],
    target_img_path: Optional[Path],
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    source_zscale: Optional[float],
    target_scale: float,
    target_zscale: Optional[float],
    source_median_filter_size: Optional[int],
    target_median_filter_size: Optional[int],
    source_clipping_quantile: Optional[float],
    target_clipping_quantile: Optional[float],
    source_gaussian_filter_sigma: Optional[float],
    target_gaussian_filter_sigma: Optional[float],
    prior_transform_path: Optional[Path],
    reverse: bool,
    scores_path: Path,
) -> None:
    mask_matching_algorithm_types: list[
        type[MaskMatchingAlgorithm]
    ] = plugin_manager.hook.spellmatch_get_mask_matching_algorithm(
        name=mask_matching_algorithm_name
    )
    if len(mask_matching_algorithm_types) > 1:
        raise SpellmatchException(
            "Discovered multiple mask matching algorithms for name "
            f"'{mask_matching_algorithm_name}': {mask_matching_algorithm_types}"
        )
    assert len(mask_matching_algorithm_types) == 1
    mask_matching_algorithm_type = mask_matching_algorithm_types[0]
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
        and (prior_transform_path is None or prior_transform_path.is_file())
        and (not scores_path.exists() or scores_path.is_file())
    ):
        source_mask_files = [source_mask_path]
        target_mask_files = [target_mask_path]
        source_img_files = [source_img_path]
        target_img_files = [target_img_path]
        prior_transform_files = [prior_transform_path]
        scores_files = [scores_path]
    elif (
        source_mask_path.is_dir()
        and target_mask_path.is_dir()
        and (source_img_path is None or source_img_path.is_dir())
        and (target_img_path is None or target_img_path.is_dir())
        and (prior_transform_path is None or prior_transform_path.is_dir())
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
        if prior_transform_path is not None:
            prior_transform_files = glob_sorted(
                prior_transform_path, "*.npy", expect=len(source_mask_files)
            )
        else:
            prior_transform_files = [None] * len(source_mask_files)
        scores_path.mkdir(exist_ok=True)
        scores_files = [
            scores_path
            / f"scores_{source_mask_file.stem}_to_{target_mask_file.stem}.nc"
            for source_mask_file, target_mask_file in zip(
                source_mask_files, target_mask_files
            )
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
        prior_transform_file,
        scores_file,
    ) in enumerate(
        zip(
            source_mask_files,
            target_mask_files,
            source_img_files,
            target_img_files,
            prior_transform_files,
            scores_files,
        )
    ):
        if len(source_mask_files) > 1:
            logger.info(f"MASK PAIR {i + 1}/{len(source_mask_files)}")
        source_mask = io.read_mask(
            source_mask_file, scale=source_scale, zscale=source_zscale
        )
        logger.info(
            f"Source mask: {source_mask_file.name} ({describe_mask(source_mask)})"
        )
        target_mask = io.read_mask(
            target_mask_file, scale=target_scale, zscale=target_zscale
        )
        logger.info(
            f"Target mask: {target_mask_file.name} ({describe_mask(target_mask)})"
        )
        if source_img_file is not None:
            source_img = io.read_image(
                source_img_file,
                panel=source_panel,
                scale=source_scale,
                zscale=source_zscale,
            )
            preprocess_image(
                source_img,
                median_filter_size=source_median_filter_size,
                clipping_quantile=source_clipping_quantile,
                gaussian_filter_sigma=source_gaussian_filter_sigma,
                inplace=True,
            )
            logger.info(
                f"Source image: {source_img_file.name} ({describe_image(source_img)})"
            )
        else:
            source_img = None
            logger.info("Source image: None")
        if target_img_file is not None:
            target_img = io.read_image(
                target_img_file,
                panel=target_panel,
                scale=target_scale,
                zscale=target_zscale,
            )
            preprocess_image(
                target_img,
                median_filter_size=target_median_filter_size,
                clipping_quantile=target_clipping_quantile,
                gaussian_filter_sigma=target_gaussian_filter_sigma,
                inplace=True,
            )
            logger.info(
                f"Target image: {target_img_file.name} ({describe_image(target_img)})"
            )
        else:
            target_img = None
            logger.info("Target image: None")
        if prior_transform_file is not None:
            prior_transform = io.read_transform(prior_transform_file)
            logger.info(
                f"Prior transform: {prior_transform_file.name} "
                f"({describe_transform(prior_transform)})"
            )
        else:
            prior_transform = None
            logger.info("Prior transform: None")
        if reverse:
            source_mask, target_mask = target_mask, source_mask
            source_img, target_img = target_img, source_img
            if prior_transform is not None:
                prior_transform_matrix = np.linalg.inv(prior_transform.params)
                prior_transform = ProjectiveTransform(matrix=prior_transform_matrix)
        info, scores = mask_matching_algorithm.match_masks(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            prior_transform=prior_transform,
        )
        if reverse:
            scores = scores.transpose()
        io.write_scores(scores_file, scores)
        logger.info(f"Info: {info}")
        logger.info(f"Scores: {scores_file.name} ({describe_scores(scores)})")


@cli.command(
    name="assign",
    help="Assign cells (convert cell alignment scores to cell assignments)",
)
@click.argument(
    "scores_path",
    metavar="SCORES",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--reverse-scores",
    "reverse_scores_path",
    type=click.Path(exists=True, path_type=Path),
    help="Target-to-source cell alignment scores file",
)
@click.option(
    "--normalize/--no-normalize",
    "normalize",
    default=False,
    show_default=True,
    help="Rescale alignment scores of each cell to sum up to 1.0",
)
@click.option(
    "--min-score",
    "min_score",
    type=click.FloatRange(min=0, min_open=True),
    help="Minimum cell alignment score",
)
@click.option(
    "--min-scoreQ",
    "min_score_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Minimum cell alignment score quantile",
)
@click.option(
    "--margin-thres",
    "margin_thres",
    type=click.FloatRange(min=0, min_open=True),
    help="Cell alignment score margin threshold",
)
@click.option(
    "--margin-thresQ",
    "margin_thres_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Cell alignment score margin threshold quantile",
)
@click.option(
    "--max/--no-max",
    "max_assignment",
    default=False,
    show_default=True,
    help="Enable maximum cell alignment score assignment",
)
@click.option(
    "--linear-sum/--no-linear-sum",
    "linear_sum_assignment",
    default=False,
    show_default=True,
    help="Enable linear sum assignment",
)
@click.option(
    "--min-post-score",
    "min_post_assignment_score",
    type=click.FloatRange(min=0, min_open=True),
    help="Minimum cell alignment score after cell assignment",
)
@click.option(
    "--min-post-scoreQ",
    "min_post_assignment_score_quantile",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    help="Minimum cell alignment score quantile after cell assignment",
)
@click.option(
    "--direction",
    "assignment_direction",
    required=True,
    type=click.Choice([item.value for item in AssignmentDirection]),
    help="Direction of the generated cell assignment",
)
@click.option(
    "--source-mask",
    "--source-masks",
    "source_mask_path",
    type=click.Path(exists=True, path_type=Path),
    help="Source mask file(s)",
)
@click.option(
    "--target-mask",
    "--target-masks",
    "target_mask_path",
    type=click.Path(exists=True, path_type=Path),
    help="Target mask file(s)",
)
@click.option(
    "--source-scale",
    "source_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (all axes)",
)
@click.option(
    "--source-zscale",
    "source_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Source pixel size (z-axis)",
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (all axes)",
)
@click.option(
    "--target-zscale",
    "target_zscale",
    type=click.FloatRange(min=0, min_open=True),
    help="Target pixel size (z-axis)",
)
@click.option(
    "--show",
    "show",
    type=click.IntRange(min=0, min_open=True),
    help="Enable visualization",
)
@click.option(
    "--validate",
    "validation_assignment_path",
    type=click.Path(exists=True, path_type=Path),
    help="Cell assignment file for validation",
)
@click.argument(
    "assignment_path",
    metavar="ASSIGNMENT(S)",
    type=click.Path(path_type=Path),
)
@click_log.simple_verbosity_option(logger=logger)
@catch_exception(handle=SpellmatchException)
def cli_assign(
    scores_path: Path,
    reverse_scores_path: Optional[Path],
    normalize: bool,
    min_score: Optional[float],
    min_score_quantile: Optional[float],
    margin_thres: Optional[float],
    margin_thres_quantile: Optional[float],
    max_assignment: bool,
    linear_sum_assignment: bool,
    min_post_assignment_score: Optional[float],
    min_post_assignment_score_quantile: Optional[float],
    assignment_direction: str,
    source_mask_path: Optional[Path],
    target_mask_path: Optional[Path],
    source_scale: float,
    source_zscale: Optional[float],
    target_scale: float,
    target_zscale: Optional[float],
    show: Optional[int],
    validation_assignment_path: Optional[Path],
    assignment_path: Path,
) -> None:
    if (
        scores_path.is_file()
        and (reverse_scores_path is None or reverse_scores_path.is_file())
        and (source_mask_path is None or source_mask_path.is_file())
        and (target_mask_path is None or target_mask_path.is_file())
        and (validation_assignment_path is None or validation_assignment_path.is_file())
        and (not assignment_path.exists() or assignment_path.is_file())
    ):
        scores_files = [scores_path]
        reverse_scores_files = [reverse_scores_path]
        source_mask_files = [source_mask_path]
        target_mask_files = [target_mask_path]
        validation_assignment_files = [validation_assignment_path]
        assignment_files = [assignment_path]
    elif (
        scores_path.is_dir()
        and (reverse_scores_path is None or reverse_scores_path.is_dir())
        and (source_mask_path is None or source_mask_path.is_dir())
        and (target_mask_path is None or target_mask_path.is_dir())
        and (validation_assignment_path is None or validation_assignment_path.is_dir())
        and (not assignment_path.exists() or assignment_path.is_dir())
    ):
        scores_files = glob_sorted(scores_path, "*.nc")
        if reverse_scores_path is not None:
            reverse_scores_files = glob_sorted(
                reverse_scores_path, "*.nc", expect=len(scores_files)
            )
        else:
            reverse_scores_files = [None] * len(scores_files)
        if source_mask_path is not None:
            source_mask_files = glob_sorted(
                source_mask_path, "*.tiff", expect=len(scores_files)
            )
        else:
            source_mask_files = [None] * len(scores_files)
        if target_mask_path is not None:
            target_mask_files = glob_sorted(
                target_mask_path, "*.tiff", expect=len(scores_files)
            )
        else:
            target_mask_files = [None] * len(scores_files)
        validation_assignment_files = glob_sorted(
            validation_assignment_path, "*.csv", expect=len(scores_files)
        )
        assignment_path.mkdir(exist_ok=True)
        assignment_files = [
            assignment_path
            / scores_file.with_suffix(".csv").name.replace("scores_", "assignment_")
            for scores_file in scores_files
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    for i, (
        scores_file,
        reverse_scores_file,
        source_mask_file,
        target_mask_file,
        validation_assignment_file,
        assignment_file,
    ) in enumerate(
        zip(
            scores_files,
            reverse_scores_files,
            source_mask_files,
            target_mask_files,
            validation_assignment_files,
            assignment_files,
        )
    ):
        if len(scores_files) > 1:
            logger.info(f"SCORES {i + 1}/{len(scores_files)}")
        scores = io.read_scores(scores_file)
        logger.info(f"Scores: {scores_file.name} ({describe_scores(scores)})")
        if reverse_scores_file is not None:
            reverse_scores = io.read_scores(reverse_scores_file)
            logger.info(
                f"Reverse scores: {reverse_scores_file.name} "
                f"({describe_scores(reverse_scores)})"
            )
        else:
            reverse_scores = None
            logger.info("Reverse scores: None")
        if source_mask_file is not None:
            source_mask = io.read_mask(
                source_mask_file, scale=source_scale, zscale=source_zscale
            )
            logger.info(
                f"Source mask: {source_mask_file.name} ({describe_mask(source_mask)})"
            )
        else:
            source_mask = None
            logger.info("Source mask: None")
        if target_mask_file is not None:
            target_mask = io.read_mask(
                target_mask_file, scale=target_scale, zscale=target_zscale
            )
            logger.info(
                f"Target mask: {target_mask_file.name} ({describe_mask(target_mask)})"
            )
        else:
            target_mask = None
            logger.info("Target mask: None")
        if validation_assignment_file is not None:
            validation_assignment = io.read_assignment(validation_assignment_file)
        else:
            validation_assignment = None
        assignment = assign(
            scores,
            reverse_scores=reverse_scores,
            normalize=normalize,
            min_score=min_score,
            min_score_quantile=min_score_quantile,
            margin_thres=margin_thres,
            margin_thres_quantile=margin_thres_quantile,
            max_assignment=max_assignment,
            linear_sum_assignment=linear_sum_assignment,
            min_post_assignment_score=min_post_assignment_score,
            min_post_assignment_score_quantile=min_post_assignment_score_quantile,
            assignment_direction=assignment_direction,
        )
        if show is not None and source_mask is not None and target_mask is not None:
            show_assignment(source_mask, target_mask, assignment, n=show)
        io.write_assignment(assignment_file, assignment)
        logger.info(
            f"Assignment: {assignment_file.name} ({describe_assignment(assignment)})"
        )
        if validation_assignment is not None:
            recovered = validate_assignment(assignment, validation_assignment)
            logger.info(
                f"Validation: {validation_assignment_file.name} "
                f"({describe_assignment(validation_assignment, recovered=recovered)})"
            )


if __name__ == "__main__":
    cli()
