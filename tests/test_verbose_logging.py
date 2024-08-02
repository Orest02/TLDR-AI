import pytest
from click.testing import CliRunner
from cli import main_cli

def test_cli_verbose_logging(caplog):
    runner = CliRunner()
    result = runner.invoke(main_cli, ['apply function to pandas column', '-v'])

    assert result.exit_code == 0
    assert any(record.levelname == "DEBUG" for record in caplog.records)
