from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import json
import re
import time
import math

import requests
from bs4 import BeautifulSoup
from .storage import read_json, write_json

BASE_URL = "https://books.toscrape.com/"
USER_AGENT = "EducationalKnowledgeSourceScraper/1.0 (academic research; respectful crawling)"


def allowed_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.netloc == "books.toscrape.com"
            and not parsed.username and ".." not in parsed.path.split("/"))


@dataclass
class RawBook:
    source_url: str
    title: str
    category_raw: str
    description_raw: str
    price_raw: str
    rating_raw: str
    availability_raw: str
    upc_raw: str
    collected_at: str
    source_mode: str


def _text(element) -> str:
    return " ".join(element.get_text(" ", strip=True).split()) if element else ""


def parse_listing(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("article.product_pod h3 a[href]")
    if not cards:
        raise ValueError(f"The listing contains no product links: {page_url}")
    return [urljoin(page_url, a["href"]) for a in cards]


def parse_book(html: str, source_url: str, mode: str) -> RawBook:
    soup = BeautifulSoup(html, "html.parser")
    title = _text(soup.select_one(".product_main h1"))
    if not title:
        raise ValueError(f"Cannot find book title: {source_url}")

    breadcrumb = [_text(li) for li in soup.select(".breadcrumb li")]
    category = breadcrumb[-2] if len(breadcrumb) >= 3 else ""
    description_header = soup.select_one("#product_description")
    description = _text(description_header.find_next_sibling("p")) if description_header else ""
    star = soup.select_one(".product_main p.star-rating")
    rating = next((c for c in star.get("class", []) if c != "star-rating"), "") if star else ""
    table = {
        _text(tr.select_one("th")): _text(tr.select_one("td"))
        for tr in soup.select("table.table.table-striped tr")
    }
    price = _text(soup.select_one(".product_main .price_color")) or table.get("Price (incl. tax)", "")
    availability = table.get("Availability", "") or _text(soup.select_one(".product_main .availability"))

    return RawBook(
        source_url=source_url,
        title=title,
        category_raw=category,
        description_raw=description,
        price_raw=price,
        rating_raw=rating,
        availability_raw=availability,
        upc_raw=table.get("UPC", ""),
        collected_at=datetime.now(timezone.utc).isoformat(),
        source_mode=mode,
    )


class DemoFetcher:
    """Load synthetic HTML fixtures without network access."""
    def __init__(self, fixtures: Path, html_cache: Path):
        self.fixtures = fixtures
        self.html_cache = html_cache
        self.html_cache.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> str:
        parsed = urlparse(url)
        if not allowed_url(url):
            raise ValueError(f"Unexpected host: {url}")
        path = self.fixtures / parsed.path.lstrip("/")
        if not path.is_file():
            raise FileNotFoundError(f"No demo fixture: {path}")
        html = path.read_text(encoding="utf-8")
        self.html_cache.joinpath(sha256(url.encode()).hexdigest()[:16] + ".html").write_text(html, encoding="utf-8")
        return html


class LiveFetcher:
    """Rate-limited HTTP with robots.txt checks and HTML snapshots."""
    def __init__(self, html_cache: Path, delay: float = 1.5):
        if not math.isfinite(delay) or delay < 1:
            raise ValueError("Live requests require a delay of at least 1 second")
        self.html_cache = html_cache
        self.html_cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.last_fetch = None
        self.delay = delay
        self.robot = RobotFileParser()
        robots_url = urljoin(BASE_URL, "robots.txt")
        response = self._request(robots_url)
        if response.status_code == 200:
            self.robot.parse(response.text.splitlines())
        elif response.status_code in (404, 410):
            self.robot.allow_all = True
        else:
            raise RuntimeError(f"Cannot establish robots.txt policy: HTTP {response.status_code}")
        crawl_delay = self.robot.crawl_delay(USER_AGENT.split("/", 1)[0])
        if crawl_delay:
            self.delay = max(self.delay, float(crawl_delay))
        rate = self.robot.request_rate(USER_AGENT.split("/", 1)[0])
        if rate and rate.requests:
            self.delay = max(self.delay, rate.seconds / rate.requests)

    def _request(self, url):
        for attempt in range(2):
            if self.last_fetch is not None:
                time.sleep(max(0, self.delay - (time.monotonic() - self.last_fetch)))
            try:
                response = self.session.get(url, timeout=20, allow_redirects=False)
            except (requests.ConnectionError, requests.Timeout):
                if attempt:
                    raise
                continue
            finally:
                self.last_fetch = time.monotonic()
            if response.status_code in (502, 503, 504) and attempt == 0:
                continue
            if 300 <= response.status_code < 400:
                raise RuntimeError("Redirect rejected: only the allowed source is supported")
            return response

    def get(self, url: str) -> str:
        if not allowed_url(url):
            raise ValueError(f"External URL rejected: {url}")
        if not self.robot.can_fetch(USER_AGENT.split("/", 1)[0], url):
            raise PermissionError(f"Disallowed by robots.txt: {url}")
        response = self._request(url)
        response.raise_for_status()
        response.encoding = "utf-8"
        html = response.text
        self.html_cache.joinpath(sha256(url.encode()).hexdigest()[:16] + ".html").write_text(html, encoding="utf-8")
        return html


def scrape(source: str, limit: int, output: Path, fixture_root: Path, delay: float = 1.5,
           progress=None, resume: bool = False) -> list[dict]:
    if source not in ("live", "demo"):
        raise ValueError("source must be live or demo")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if output.exists() and not resume:
        raise FileExistsError("Raw snapshot already exists; use a new run or explicit resume")
    books = read_json(output) if resume and output.exists() else []
    if not isinstance(books, list) or any(b.get("source_mode") != source for b in books):
        raise ValueError("Checkpoint source mismatch")
    write_json(output, books)
    errors_path = output.parent / "scrape_errors.json"
    errors = read_json(errors_path) if resume and errors_path.exists() else []
    log = progress or (lambda message: None)
    html_cache = output.parent / "html_snapshots"
    log(f"Источник {source}: проверка доступа и подготовка сбора")
    fetcher = DemoFetcher(fixture_root, html_cache) if source == "demo" else LiveFetcher(html_cache, delay)
    seen = {b["source_url"] for b in books}
    visited = set()
    page = 1

    def fetch(url):
        if not allowed_url(url):
            raise ValueError("External source URL rejected")
        cached = html_cache / (sha256(url.encode()).hexdigest()[:16] + ".html")
        if resume and cached.exists():
            return cached.read_text(encoding="utf-8")
        return fetcher.get(url)

    while len(books) < limit:
        listing_url = urljoin(BASE_URL, f"catalogue/page-{page}.html")
        if source == "demo" and page > 1:
            break
        log(f"Страница {page}/до 50; собрано {len(books)}/{limit}")
        try:
            html = fetch(listing_url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404 and page > 1:
                break
            raise
        for url in parse_listing(html, listing_url):
            if url in seen or url in visited:
                continue
            visited.add(url)
            log(f"Книга {len(books) + 1}/{limit}: {url}")
            html_book = fetch(url)
            try:
                book = asdict(parse_book(html_book, url, source))
            except ValueError:
                errors.append({"source_url": url, "error": "invalid_book_html"})
                write_json(errors_path, errors)
                log("Ошибка HTML карточки; запись пропущена и сохранена в scrape_errors.json")
                continue
            books.append(book)
            seen.add(url)
            write_json(output, books)
            if len(books) >= limit:
                break
        page += 1
        if page > 50:
            break
    write_json(errors_path, errors)
    log(f"Сбор завершён: {len(books)} записей, ошибок HTML: {len(errors)}")
    return books
