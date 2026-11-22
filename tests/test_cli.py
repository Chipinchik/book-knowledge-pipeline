import sys

from coursework.cli import main
from coursework.pipeline import latest_run
from coursework.storage import read_json


def test_cli_all_normalize_report_evaluate(tmp_path, monkeypatch, capsys):
    def command(*arguments):
        monkeypatch.setattr(sys, "argv", ["coursework.cli", *arguments, "--out", str(tmp_path)])
        main()
        return capsys.readouterr().out
    assert '"exported_facts": 63' in command("all", "--source", "demo", "--enrich", "rules", "--limit", "10")
    initial = latest_run(tmp_path)
    command("normalize", "--raw", str(initial / "raw_books.json"), "--enrich", "rules")
    final = latest_run(tmp_path)
    assert initial != final
    assert read_json(initial / "manifest.json")["raw_sha256"] == read_json(final / "manifest.json")["raw_sha256"]
    assert '"llm_processed": 0' in command("report")
    assert '"topic_accuracy": null' in command("evaluate")
