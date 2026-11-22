from unittest.mock import Mock
import json
from streamlit.testing.v1 import AppTest
from coursework.llm_provider import ProviderConfig, GeminiProvider
from coursework.pipeline import ROOT, execute
from coursework.experiments import prepare_experiment, run_method


def app_for(tmp_path, monkeypatch):
    monkeypatch.setenv('BOOK_PIPELINE_OUTPUT', str(tmp_path))
    return AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()


def widget(items, label):
    return next(b for b in items if b.label == label)


def page(app, name):
    return widget(app.radio, 'Раздел').set_value(name).run()


def technical(app):
    return widget(app.checkbox, 'Показать короткие проверки и исходные наборы').check().run()


def test_streamlit_render_filter_and_export(tmp_path, monkeypatch):
    execute(tmp_path, progress=lambda _: None)
    app = technical(app_for(tmp_path, monkeypatch))
    assert not app.exception
    assert any('синтетические' in w.value for w in app.warning)
    page(app, 'Каталог книг')
    assert len(app.dataframe[0].value) == 9
    widget(app.text_input, 'Поиск по названию').set_value('HARBOR').run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    widget(app.multiselect, 'Темы').set_value([]).run()
    assert len(app.dataframe[0].value) == 0
    page(app, 'Аналитика')
    assert not app.exception
    assert len(app.get('download_button')) >= 6
    page(app, 'Экспорт и эксперименты')
    assert not app.exception
    assert len(app.get('download_button')) >= 16


def test_empty_dashboard_collect_and_process_from_ui(tmp_path, monkeypatch):
    app = app_for(tmp_path, monkeypatch)
    assert not app.exception and not app.metric
    widget(app.checkbox, 'Только просмотр').uncheck().run()
    page(app, 'Сбор данных')
    widget(app.selectbox, 'Источник').set_value('demo').run()
    widget(app.button, 'Собрать данные').click().run()
    assert not app.exception
    page(app, 'Нормализация')
    widget(app.radio, 'Назначение').set_value('technical').run()
    widget(app.button, 'Подготовить общий эксперимент').click().run()
    assert not app.exception
    widget(app.button, 'Запустить Rules').click().run()
    assert not app.exception
    page(app, 'Каталог книг')
    assert len(app.dataframe[0].value) == 2


def test_gemini_requires_explicit_confirmation_in_ui(tmp_path, monkeypatch):
    raw = execute(tmp_path, limit=1, progress=lambda _: None)
    prepare_experiment(tmp_path, raw/'raw_books.json', 1, ProviderConfig(), purpose='technical')
    app = technical(app_for(tmp_path, monkeypatch))
    widget(app.checkbox, 'Только просмотр').uncheck().run()
    widget(app.radio, 'Метод').set_value('gemini').run()
    page(app, 'Нормализация')
    assert not app.exception
    assert widget(app.button, 'Запустить Gemini').disabled


def test_comparison_dashboard_http_double(tmp_path, monkeypatch):
    raw = execute(tmp_path, limit=2, progress=lambda _: None)
    answer = {'topic': 'Другое', 'keywords': [], 'summary_ru': '', 'insufficient_data': True}
    session = Mock(post=Mock(return_value=Mock(status_code=200, headers={}, json=Mock(return_value={
        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}],
        'usage': {'total_tokens': 12}}))))
    cfg = ProviderConfig(api_key='test-only-credential', free_tier_confirmed=True, delay=1)
    monkeypatch.setattr('coursework.normalize.create_provider', lambda c, ledger, run_id: GeminiProvider(c, ledger, session=session, sleep=lambda _: None, run_id=run_id))
    exp = prepare_experiment(tmp_path,raw/'raw_books.json',2,cfg,purpose='technical')
    run_method(tmp_path,exp['experiment_id'],'rules')
    run_method(tmp_path,exp['experiment_id'],'gemini',config=cfg,api_confirmed=True)
    app = technical(app_for(tmp_path, monkeypatch))
    page(app, 'Сравнение методов')
    assert not app.exception
    assert any('Accuracy и Macro-F1 не измерены' in x.value for x in app.info)
    assert session.post.call_count == 2
    monkeypatch.setattr(ProviderConfig, 'from_env', Mock(side_effect=AssertionError('offline must not read credentials')))
    widget(app.radio, 'Метод').set_value('gemini').run()
    for name in ['Главная','Каталог книг','Аналитика','Экспорт и эксперименты','Нормализация']:
        page(app,name)
        assert not app.exception
    assert session.post.call_count == 2


def test_empty_completed_run_dashboard(tmp_path, monkeypatch):
    from coursework.storage import write_json
    raw = tmp_path/'input.json'
    write_json(raw, [])
    execute(tmp_path,raw_path=raw,progress=lambda _:None)
    app = technical(app_for(tmp_path,monkeypatch))
    for name in ['Каталог книг','Аналитика','Экспорт и эксперименты']:
        page(app,name)
        assert not app.exception
