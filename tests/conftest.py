import pytest
import requests
import os


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch):
    """Block real HTTP requests in tests."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in pytest; provide a fake HTTP response")
    monkeypatch.setattr(requests.Session, "request", forbidden)
    # Ignore local credentials even when HTTP is mocked.
    monkeypatch.setattr('coursework.llm_provider.dotenv_values', lambda path: {})
    for name in list(os.environ):
        if name.startswith('GEMINI_'):
            monkeypatch.delenv(name)
