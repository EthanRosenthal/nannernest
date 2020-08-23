import click_spinner
from pathlib import Path
import typer

from nannernest import nesting, segmentation, slicing, viz
from nannernest.exceptions import SegmentNotFound


_BANANA = "\U0001F34C"
_BREAD = "\U0001F35E"
_FG = typer.colors.GREEN


app = typer.Typer()


@app.command()
def cli(
    image_path: Path = typer.Argument(
        ..., help="Image file which contains bread and banana"
    ),
    num_slices: int = typer.Option(
        22, help="Total number of slices to cut the banana into. This number defines the slice thickness."
    ),
    mask_threshold: float = typer.Option(0.6, help="Threshold of segmentation mask."),
    peel_scaler: float = typer.Option(
        0.8,
        help="Fraction of slice that is assumed to belong to banana insides versus the peel.",
    ),
    ellipse_ratio: float = typer.Option(
        0.85, help="Assumed ratio of minor axis to major axis of banana slice ellipses"
    ),
    plot_segmentation: bool = typer.Option(
        False, help="Whether or not to plot the segmentation masks"
    ),
    plot_slicing: bool = typer.Option(
        False, help="Whether or not to plot the slicing circle and skeleton"
    ),
    output: str = typer.Option("perfect_sandwich.jpg", help="Name of file to output"),
    suppress: bool = typer.Option(
        False,
        help="Whether or not to suppress the output file from being automatically opened",
    ),
):

    typer.secho(f"Segmenting image looking for {_BANANA} and {_BREAD}", fg=_FG)

    try:
        with click_spinner.spinner():
            image, banana, bread = segmentation.run(image_path)
    except SegmentNotFound:
        typer.secho(
            "Oh no! Can't find the banana or bread in the image", fg=typer.colors.RED,
        )
        raise SegmentNotFound

    typer.secho(f"Slicing the {_BANANA}", fg=_FG)
    with click_spinner.spinner():
        slices, banana_circle, banana_centroid, banana_skeleton = slicing.run(
            banana.mask,
            num_slices=num_slices,
            mask_threshold=mask_threshold,
            peel_scaler=peel_scaler,
            ellipse_ratio=ellipse_ratio,
        )

    typer.secho("Optimizing slice coverage", fg=_FG)
    with click_spinner.spinner():
        slices, bread_box = nesting.run(slices, bread, mask_threshold=mask_threshold)

    viz.plot(
        image,
        slices=slices,
        banana=banana if plot_segmentation else None,
        bread=bread if plot_segmentation else None,
        banana_skeleton=banana_skeleton if plot_slicing else None,
        banana_circle=banana_circle if plot_slicing else None,
        banana_centroid=banana_centroid if plot_slicing else None,
        bread_box=bread_box if plot_slicing else None,
        output=output,
    )

    typer.secho(f"Nested nanner saved to {output}", fg=_FG)
    if not suppress:
        typer.launch(output)
