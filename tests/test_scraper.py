from unittest.mock import Mock

import pytest
import requests

from coursework import scraper
from coursework.pipeline import ROOT, execute
from coursework.storage import read_json


def http(status=200, text=""):
    result = Mock(status_code=status, text=text)
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


def session_for(monkeypatch, replies):
    session = Mock()
    session.headers = {}
    session.get.side_effect = replies
    monkeypatch.setattr(scraper.requests, "Session", lambda: session)
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: None)
    return session


def test_robots_rate_limit_redirects_and_allowlist(tmp_path, monkeypatch):
    session = session_for(monkeypatch, [http(text="User-agent: *\nDisallow: /private\nCrawl-delay: 3\nRequest-rate: 1/5"), http(text="ok")])
    fetcher = scraper.LiveFetcher(tmp_path, 1)
    assert fetcher.delay == 5
    assert fetcher.last_fetch is not None  # robots also counts toward the rate limit
    assert "User-Agent" in session.headers
    for url in ("https://evil.example/a", "http://books.toscrape.com/a", "https://books.toscrape.com@evil.example/a", "https://["):
        with pytest.raises(ValueError):
            fetcher.get(url)
    with pytest.raises(PermissionError):
        fetcher.get(scraper.BASE_URL + "private/a")
    assert fetcher.get(scraper.BASE_URL + "ok") == "ok"
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("status", [301, 403, 429, 500])
def test_robots_fail_closed(tmp_path, monkeypatch, status):
    session_for(monkeypatch, [http(status)])
    with pytest.raises(RuntimeError):
        scraper.LiveFetcher(tmp_path)


def test_transient_retry_only_and_redirect_rejected(tmp_path, monkeypatch):
    session = session_for(monkeypatch, [http(404), requests.Timeout(), http(text="book"), http(302)])
    fetcher = scraper.LiveFetcher(tmp_path)
    assert fetcher.get(scraper.BASE_URL + "a") == "book"
    with pytest.raises(RuntimeError, match="Redirect"):
        fetcher.get(scraper.BASE_URL + "b")
    assert session.get.call_count == 4


def test_interrupt_and_resume_preserves_collected_data(tmp_path, monkeypatch):
    original = scraper.DemoFetcher.get
    def interrupted(self, url):
        if "book_3/" in url:
            raise KeyboardInterrupt()
        return original(self, url)
    monkeypatch.setattr(scraper.DemoFetcher, "get", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, source="demo", limit=5, progress=lambda _: None)
    run = next((tmp_path / "runs").iterdir())
    before = read_json(run / "raw_books.json")
    assert len(before) == 2
    assert read_json(run / "manifest.json")["status"] == "interrupted"
    monkeypatch.setattr(scraper.DemoFetcher, "get", original)
    execute(tmp_path, resume=run, progress=lambda _: None)
    after = read_json(run / "raw_books.json")
    assert len(after) == 5 and after[:2] == before
    with pytest.raises(ValueError, match="незавершённый"):
        execute(tmp_path, resume=run)


def test_invalid_html_record_is_logged(tmp_path, monkeypatch):
    original = scraper.DemoFetcher.get
    monkeypatch.setattr(scraper.DemoFetcher, "get", lambda self, url: "<html></html>" if "book_1/" in url else original(self, url))
    rows = scraper.scrape("demo", 2, tmp_path / "raw.json", ROOT / "fixtures")
    assert len(rows) == 2
    assert read_json(tmp_path / "scrape_errors.json")[0]["error"] == "invalid_book_html"


@pytest.mark.parametrize("delay", [0, 0.9, float("nan"), float("inf")])
def test_invalid_delay(delay, tmp_path):
    with pytest.raises(ValueError):
        scraper.LiveFetcher(tmp_path, delay)
