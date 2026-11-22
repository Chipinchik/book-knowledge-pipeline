from pathlib import Path
import json

from coursework.scraper import parse_listing, parse_book, scrape
from coursework.normalize import parse_price, parse_stock, topic_from_rules, normalize, KnowledgeRecord
from coursework.report import generate_report

ROOT = Path(__file__).resolve().parents[1]


def test_parsing_fixture_html():
    listing = (ROOT / 'fixtures/catalogue/page-1.html').read_text(encoding='utf-8')
    urls = parse_listing(listing, 'https://books.toscrape.com/catalogue/page-1.html')
    assert len(urls) == 10
    html = (ROOT / 'fixtures/catalogue/book_1/index.html').read_text(encoding='utf-8')
    book = parse_book(html, urls[0], 'demo')
    assert book.title == 'Night at the Harbor'
    assert book.category_raw == 'Mystery'
    assert book.upc_raw == 'b000000000000001'
    assert 'detective' in book.description_raw


def test_primitive_normalization():
    assert parse_price('£51.77') == 51.77
    assert parse_price('') is None
    assert parse_stock('In stock (22 available)') == (True, 22)
    assert parse_stock('Out of stock') == (False, 0)
    assert topic_from_rules('Mystery') == 'Детектив'
    assert topic_from_rules('Romance') == 'Другое'


def test_end_to_end_demo(tmp_path: Path):
    raw_path = tmp_path / 'raw.json'
    norm_path = tmp_path / 'normalized.json'
    raw = scrape('demo', 10, raw_path, ROOT / 'fixtures')
    assert len(raw) == 10
    normalized = normalize(raw_path, norm_path)
    assert len(normalized) == 9  # Duplicate UPC in synthetic fixture.
    for record in normalized:
        KnowledgeRecord.model_validate(record)
    assert any('missing_description' in b['quality_flags'] for b in normalized)
    metrics = generate_report(raw_path, norm_path, tmp_path / 'reports')
    assert metrics['raw_records'] == 10
    assert metrics['removed_duplicates'] == 1
    assert metrics['normalized_records'] == 9
    assert metrics['llm_processed'] == 0
    assert metrics['completeness_pct']['description_raw'] < 100
    assert (tmp_path / 'reports' / 'knowledge_source.csv').exists()
    assert (tmp_path / 'reports' / 'topics.png').exists()
    assert (tmp_path / 'reports' / 'facts_triples.csv').exists()
    assert (tmp_path / 'reports' / 'knowledge_source.sqlite').exists()
    assert metrics["exported_facts"] > len(normalized)
