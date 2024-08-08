from click.testing import CliRunner

from tldrai.cli import main_cli


def test_cli_integration():
    runner = CliRunner()
    result = runner.invoke(
        main_cli,
        [
            "apply function to pandas column",
            "--set",
            "summarization_pipeline.model=stable-code",
        ],
    )

    assert result.exit_code == 0
    # There's more than one word in the output
    assert len(result.output.split()) > 1
