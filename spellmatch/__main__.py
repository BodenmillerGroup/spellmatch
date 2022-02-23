from pathlib import Path
from typing import Any, Optional, Type

import click
import click_log
import pluggy
import yaml
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)

from . import hookspecs, io
from ._spellmatch import logger as root_logger
from .matching.algorithms import icp
from .registration import automatic as automatic_registration
from .registration import interactive as interactive_registration
from .registration.automatic.metrics import (
    metric_types as automatic_registration_metric_types,
)
from .registration.automatic.optimizers import (
    optimizer_types as automatic_registration_optimizer_types,
)

transform_types: dict[str, Type[ProjectiveTransform]] = {
    "euclidean": EuclideanTransform,
    "similarity": SimilarityTransform,
    "affine": AffineTransform,
}

click_log.basic_config(logger=root_logger)


def get_plugin_manager() -> pluggy.PluginManager:
    pm = pluggy.PluginManager("spellmatch")
    pm.add_hookspecs(hookspecs)
    pm.load_setuptools_entrypoints("spellmatch")
    pm.register(icp, name="spellmatch-icp")
    return pm


@click.group()
@click.version_option()
def cli() -> None:
    pass


@cli.command()
@click.argument(
    "source_mask_file",
    metavar="SOURCE_MASK",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "target_mask_file",
    metavar="TARGET_MASK",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--source-image",
    "source_img_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--target-image",
    "target_img_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
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
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--transform-type",
    "transform_type_name",
    default="affine",
    show_default=True,
    type=click.Choice(list(transform_types.keys())),
)
@click.argument(
    "cell_pairs_file",
    metavar="CELL_PAIRS",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.argument(
    "transform_file",
    metavar="TRANSFORM",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click_log.simple_verbosity_option(logger=root_logger)
def align(
    source_mask_file: Path,
    target_mask_file: Path,
    source_img_file: Optional[Path],
    target_img_file: Optional[Path],
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    target_scale: float,
    transform_type_name: str,
    cell_pairs_file: Path,
    transform_file: Path,
) -> None:
    source_mask = io.read_mask(source_mask_file, scale=source_scale)
    target_mask = io.read_mask(target_mask_file, scale=target_scale)
    source_img = None
    if source_img_file is not None:
        source_panel = None
        if source_panel_file.exists():
            source_panel = io.read_panel(source_panel_file)
        source_img = io.read_image(
            source_img_file, panel=source_panel, scale=source_scale
        )
    target_img = None
    if target_img_file is not None:
        target_panel = None
        if target_panel_file.exists():
            target_panel = io.read_panel(target_panel_file)
        target_img = io.read_image(
            target_img_file, panel=target_panel, scale=target_scale
        )
    cell_pairs = None
    if cell_pairs_file.exists():
        cell_pairs = io.read_cell_pairs(cell_pairs_file)
    result = interactive_registration.align_masks(
        source_mask,
        target_mask,
        source_img=source_img,
        target_img=target_img,
        transform_type=transform_types[transform_type_name],
        cell_pairs=cell_pairs,
    )
    if result is not None:
        cell_pairs, transform = result
        io.write_cell_pairs(cell_pairs_file, cell_pairs)
        if transform is not None:
            io.write_transform(transform_file, transform)
    else:
        raise click.Abort()


@cli.command()
@click.argument(
    "source_img_path",
    metavar="SOURCE_IMAGES",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "target_img_path",
    metavar="TARGET_IMAGES",
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
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
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
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--denoise-target",
    "denoise_target",
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--blur-source",
    "blur_source",
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--blur-target",
    "blur_target",
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--metric",
    "metric_name",
    default="correlation",
    show_default=True,
    type=click.Choice(list(automatic_registration_metric_types.keys())),
)
@click.option(
    "--metric-args",
    "metric_kwargs_str",
    default="",
    show_default=True,
    type=click.STRING,
)
@click.option(
    "--optimizer",
    "optimizer_name",
    default="regular_step_gradient_descent",
    show_default=True,
    type=click.Choice(list(automatic_registration_optimizer_types.keys())),
)
@click.option(
    "--optimizer-args",
    "optimizer_kwargs_str",
    default="lr=2.0,min_step=1e-4,num_iter=500,grad_magnitude_tol=1e-8",
    show_default=True,
    type=click.STRING,
)
@click.option(
    "--transform-type",
    "sitk_transform_type_name",
    default="affine",
    show_default=True,
    type=click.Choice(list(automatic_registration.sitk_transform_types.keys())),
)
@click.option(
    "--initial-transforms",
    "initial_transform_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "transform_path",
    metavar="TRANSFORMS",
    type=click.Path(path_type=Path),
)
@click_log.simple_verbosity_option(logger=root_logger)
def register(
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
    metric_name: str,
    metric_kwargs_str: str,
    optimizer_name: str,
    optimizer_kwargs_str: str,
    sitk_transform_type_name: str,
    initial_transform_path: Optional[Path],
    transform_path: Path,
) -> None:
    metric_type = automatic_registration_metric_types[metric_name]
    metric_kwargs = _parse_kwargs(metric_kwargs_str)
    metric = metric_type(**metric_kwargs)
    optimizer_type = automatic_registration_optimizer_types[optimizer_name]
    optimizer_kwargs = _parse_kwargs(optimizer_kwargs_str)
    optimizer = optimizer_type(**optimizer_kwargs)
    sitk_transform_type = automatic_registration.sitk_transform_types[
        sitk_transform_type_name
    ]
    if (
        source_img_path.is_file()
        and target_img_path.is_file()
        and (initial_transform_path is None or initial_transform_path.is_file())
    ):
        source_img_files = [source_img_path]
        target_img_files = [target_img_path]
        initial_transform_files = [initial_transform_path]
        transform_files = [transform_path]
    elif (
        source_img_path.is_dir()
        and target_img_path.is_dir()
        and (initial_transform_path is None or initial_transform_path.is_dir())
    ):
        source_img_files = sorted(source_img_path.glob("*.tiff"))
        target_img_files = sorted(target_img_path.glob("*.tiff"))
        if len(target_img_files) != len(source_img_files):
            raise click.UsageError(
                f"Expected {len(source_img_files)} target images, "
                f"found {len(target_img_files)}"
            )
        if initial_transform_path is not None:
            initial_transform_files = sorted(initial_transform_path.glob(".npy"))
            if len(initial_transform_files) != len(source_img_files):
                raise click.UsageError(
                    f"Expected {len(source_img_files)} initial transforms, "
                    f"found {len(initial_transform_files)}"
                )
        else:
            initial_transform_files = [None] * len(source_img_files)
        transform_path.mkdir(exist_ok=True)
        transform_files = [
            transform_path / source_img_file.with_suffix(".npy")
            for source_img_file in source_img_files
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    source_panel = None
    if source_panel_file.exists():
        source_panel = io.read_panel(source_panel_file)
    target_panel = None
    if target_panel_file.exists():
        target_panel = io.read_panel(target_panel_file)
    for source_img_file, target_img_file, initial_transform_file, transform_file in zip(
        source_img_files, target_img_files, initial_transform_files, transform_files
    ):
        click.echo(f"SOURCE IMAGE: {source_img_file.name}")
        click.echo(f"TARGET IMAGE: {target_img_file.name}")
        if initial_transform_file is not None:
            click.echo(f"INITIAL TRANFORM: {initial_transform_file.name}")
        else:
            click.echo("INITIAL TRANSFORM: None")
        click.echo(f"TRANSFORM OUT: {transform_file.name}")
        source_img = io.read_image(
            source_img_file, panel=source_panel, scale=source_scale
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
                            f"Source channel {source_channel} is not a channel name"
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
                            f"Target channel {target_channel} is not a channel name"
                            f"or valid index in target image {target_img_file.name}"
                        )
            else:
                raise click.UsageError(
                    "No channel specified "
                    f"for multi-channel target image {target_img_file.name}"
                )
        initial_transform = None
        if initial_transform_file is not None:
            initial_transform = io.read_transform(initial_transform_file)
        transform = automatic_registration.register_images(
            source_img,
            target_img,
            metric,
            optimizer,
            sitk_transform_type=sitk_transform_type,
            initial_transform=initial_transform,
            denoise_source=denoise_source,
            denoise_target=denoise_target,
            blur_source=blur_source,
            blur_target=blur_target,
        )
        io.write_transform(transform_file, transform)


@cli.command()
@click.argument(
    "source_mask_path",
    metavar="SOURCE_MASKS",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "target_mask_path",
    metavar="TARGET_MASKS",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source-images",
    "source_img_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
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
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--target-scale",
    "target_scale",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--algorithm",
    "matching_algorithm_name",
    default="icp",
    show_default=True,
    type=click.STRING,
)
@click.option(
    "--algorithm-args",
    "matching_algorithm_kwargs_str",
    default="max_iter=10,top_k_estim=50,max_dist=5.0",
    show_default=True,
    type=click.STRING,
)
@click.option(
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
@click_log.simple_verbosity_option(logger=root_logger)
def match(
    source_mask_path: Path,
    target_mask_path: Path,
    source_img_path: Optional[Path],
    target_img_path: Optional[Path],
    source_panel_file: Path,
    target_panel_file: Path,
    source_scale: float,
    target_scale: float,
    matching_algorithm_name: str,
    matching_algorithm_kwargs_str: str,
    transform_path: Optional[Path],
    reverse: bool,
    scores_path: Path,
) -> None:
    pm = get_plugin_manager()
    matching_algorithm_type = pm.hook.spellmatch_get_matching_algorithm(
        name=matching_algorithm_name
    )
    matching_algorithm_kwargs = _parse_kwargs(matching_algorithm_kwargs_str)
    matching_algorithm = matching_algorithm_type(**matching_algorithm_kwargs)
    if (
        source_mask_path.is_file()
        and target_mask_path.is_file()
        and (source_img_path is None or source_img_path.is_file())
        and (target_img_path is None or target_img_path.is_file())
        and (transform_path is None or transform_path.is_file())
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
    ):
        source_mask_files = sorted(source_mask_path.glob("*.tiff"))
        target_mask_files = sorted(target_mask_path.glob("*.tiff"))
        if len(target_mask_files) != len(source_mask_files):
            raise click.UsageError(
                f"Expected {len(source_mask_files)} target masks, "
                f"found {len(target_mask_files)}"
            )
        if source_img_path is not None:
            source_img_files = sorted(source_img_path.glob("*.tiff"))
            if len(source_img_files) != len(source_mask_files):
                raise click.UsageError(
                    f"Expected {len(source_mask_files)} source images, "
                    f"found {len(source_img_files)}"
                )
        else:
            source_img_files = [None] * len(source_mask_files)
        if target_img_path is not None:
            target_img_files = sorted(target_img_path.glob("*.tiff"))
            if len(target_img_files) != len(target_mask_files):
                raise click.UsageError(
                    f"Expected {len(target_mask_files)} target images, "
                    f"found {len(target_img_files)}"
                )
        else:
            target_img_files = [None] * len(target_mask_files)
        if transform_path is not None:
            transform_files = sorted(transform_path.glob(".npy"))
            if len(transform_files) != len(source_mask_files):
                raise click.UsageError(
                    f"Expected {len(source_mask_files)} transforms, "
                    f"found {len(transform_files)}"
                )
        else:
            transform_files = [None] * len(source_mask_files)
        scores_path.mkdir(exist_ok=True)
        scores_files = [
            scores_path / source_mask_file.with_suffix(".npy")
            for source_mask_file in source_mask_files
        ]
    else:
        raise click.UsageError(
            "Either specify individual files, or directories, but not both"
        )
    source_panel = None
    if source_panel_file.exists():
        source_panel = io.read_panel(source_panel_file)
    target_panel = None
    if target_panel_file.exists():
        target_panel = io.read_panel(target_panel_file)
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
        click.echo(f"SOURCE MASK: {source_mask_file.name}")
        click.echo(f"TARGET MASK: {target_mask_file.name}")
        if source_img_file is not None:
            click.echo(f"SOURCE IMAGE: {source_img_file.name}")
        else:
            click.echo("SOURCE IMAGE: None")
        if target_img_file is not None:
            click.echo(f"TARGET IMAGE: {target_img_file.name}")
        else:
            click.echo("TARGET IMAGE: None")
        if transform_file is not None:
            click.echo(f"TRANFORM: {transform_file.name}")
        else:
            click.echo("TRANSFORM: None")
        click.echo(f"SCORES OUT: {scores_file.name}")
        source_mask = io.read_mask(source_mask_file, scale=source_scale)
        target_mask = io.read_mask(target_mask_file, scale=target_scale)
        source_img = None
        if source_img_file is not None:
            source_img = io.read_image(
                source_img_file, panel=source_panel, scale=source_scale
            )
        target_img = None
        if target_img_file is not None:
            target_img = io.read_image(
                target_img_file, panel=target_panel, scale=target_scale
            )
        transform = None
        if transform_file is not None:
            transform = io.read_transform(transform_file)
        scores = matching_algorithm(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
            reverse=reverse,
        )
        io.write_scores(scores_file, scores)


@cli.command()
@click_log.simple_verbosity_option(logger=root_logger)
def assign() -> None:
    pass  # TODO implement assign command


def _parse_kwargs(kwargs_str: str) -> dict[str, Any]:
    key_value_pairs = [
        key_value_pair_str.split(sep="=", maxsplit=1)
        for key_value_pair_str in kwargs_str.split(sep=",")
    ]
    return yaml.load("\n".join(f"{k}: {v}" for k, v in key_value_pairs), yaml.Loader)


if __name__ == "__main__":
    cli()
