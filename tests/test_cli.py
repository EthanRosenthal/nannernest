from typer.testing import CliRunner

from nannernest import cli


def test_cli(image_path):
    result = CliRunner().invoke(
        cli.app,
        [str(image_path), "--suppress", "--plot-slicing", "--plot-segmentation"],
    )
    assert result.exit_code == 0
