from pathlib import Path
from typing import Optional

import click

from . import io
from .registration.interactive import register_masks


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
    "--source-img",
    "source_img_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--target-img",
    "target_img_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--panel",
    "panel_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "label_pairs_file",
    metavar="LABEL_PAIRS",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.argument(
    "transform_file",
    metavar="TRANSFORM",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def register(
    source_mask_file: Path,
    target_mask_file: Path,
    source_img_file: Optional[Path],
    target_img_file: Optional[Path],
    panel_file: Optional[Path],
    label_pairs_file: Path,
    transform_file: Path,
) -> None:
    source_mask = io.read_mask(source_mask_file)
    target_mask = io.read_mask(target_mask_file)
    panel = None
    if panel_file is not None:
        panel = io.read_panel(panel_file)
    source_img = None
    if source_img_file is not None:
        source_img = io.read_image(source_img_file, panel=panel)
    target_img = None
    if target_img_file is not None:
        target_img = io.read_image(target_img_file, panel=panel)
    label_pairs = None
    if label_pairs_file.exists():
        label_pairs = io.read_label_pairs(label_pairs_file)
    result = register_masks(
        source_mask,
        target_mask,
        source_img=source_img,
        target_img=target_img,
        label_pairs=label_pairs,
    )
    if result is not None:
        label_pairs, transform = result
        io.write_label_pairs(label_pairs_file, label_pairs)
        io.write_transform(transform_file, transform)
    else:
        raise click.Abort()


@cli.command()
def match():
    pass


if __name__ == "__main__":
    cli()
