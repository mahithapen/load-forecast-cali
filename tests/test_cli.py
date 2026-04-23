import pytest
from load_forecasting_cali.cli import main


def test_cli_help(capsys):
    """Test that the CLI help command works."""
    with pytest.raises(SystemExit) as e:
        # Simulate: load-forecast --help
        with pytest.MonkeyPatch().context() as m:
            m.setattr("sys.argv", ["load-forecast", "--help"])
            main()
    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "CAISO load forecasting pipeline CLI" in captured.out
