import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from coursework.status import event_metrics, event_kind, user_message
from coursework.dashboard import list_runs, load_run
from coursework.experiments import prepare_experiment
from coursework.llm_provider import ProviderConfig, GeminiProvider
from coursework.pipeline import execute
from coursework.storage import read_json, write_json, file_hash
from test_dashboard import app_for, technical, page, widget


def test_error_classification_keeps_every_event():
    events=[{'row':i,'status':'gemini','cached':i<2} for i in range(15)]
    events += [{'row':12,'status':'paused','error':'run_request_limit'} for _ in range(5)]
    original=copy.deepcopy(events)
    m=event_metrics({'calls':events})
    assert m['api_errors']==m['validation_errors']==m['processing_errors']==0
    assert m['limit_pauses']==m['original_error_events']==5
    # Only the latest event for each row counts.
    assert m['accepted_responses']==14
    events.append({'row':12,'status':'gemini','cached':False})
    m=event_metrics({'calls':events})
    assert m['accepted_responses']==15 and m['accepted_cached_responses']==2 and m['accepted_new_responses']==13
    assert original==events[:-1]


@pytest.mark.parametrize('code,kind',[
    ('http_429','api_error'),('http_503','api_error'),('network_outcome_unknown_no_automatic_retry','api_error'),
    ('invalid_json_response','validation_error'),('invalid_semantic_schema','validation_error'),
    ('run_request_limit','limit_pause'),('rolling_24h_request_limit','limit_pause'),
    ('uncertain_previous_request_no_automatic_retry','safety_pause'),('unexpected_failure','other_error')])
def test_error_categories(code,kind):
    assert event_kind({'error':code})==kind


def test_read_only_metrics_projection(tmp_path):
    directory=execute(tmp_path,limit=1,progress=lambda _:None)
    path=directory/'rules/normalization_stats.json'
    stats=read_json(path)
    stats['calls']=[{'row':0,'status':'paused','error':'run_request_limit'}]
    write_json(path,stats)
    metric_path=directory/'rules/metrics.json'
    stored=read_json(metric_path);stored['processing_errors']=1;write_json(metric_path,stored)
    hashes={p:file_hash(p) for p in directory.rglob('*') if p.is_file()}
    run=load_run(list_runs(tmp_path)[0])
    assert run['metrics']['processing_errors']==0 and run['metrics']['limit_pauses']==1
    assert run['stored_metrics']['processing_errors']==1
    assert all(file_hash(p)==h for p,h in hashes.items())


def test_explicit_new_experiment_keeps_first(tmp_path):
    source=execute(tmp_path,limit=1,progress=lambda _:None)
    row=read_json(source/'raw_books.json')[0]
    raw=tmp_path/'live-test-double.json'
    write_json(raw,[{**row,'source_mode':'live','upc_raw':str(i),'source_url':f'https://books.toscrape.com/{i}'} for i in range(15)])
    first=prepare_experiment(tmp_path,raw,15,ProviderConfig())
    folder=tmp_path/'experiments'/first['experiment_id']
    before={p:file_hash(p) for p in folder.rglob('*') if p.is_file()}
    second=prepare_experiment(tmp_path,raw,15,ProviderConfig(),allow_new=True)
    assert second['experiment_id']!=first['experiment_id']
    assert all(file_hash(p)==h for p,h in before.items())


def test_help_and_saved_completed_result_offline(tmp_path,monkeypatch):
    source=execute(tmp_path,limit=2,progress=lambda _:None)
    from coursework.experiments import run_method
    exp=prepare_experiment(tmp_path,source/'raw_books.json',2,ProviderConfig(),purpose='technical')
    run_method(tmp_path,exp['experiment_id'],'rules')
    monkeypatch.setattr(ProviderConfig,'from_env',Mock(side_effect=AssertionError('offline')))
    app=technical(app_for(tmp_path,monkeypatch))
    page(app,'Нормализация')
    assert not app.exception and any('уже выполнен' in x.value for x in app.info)
    assert any(b.label=='Посмотреть результаты' for b in app.button)
    page(app,'Помощь')
    assert not app.exception
    assert any('Как работает кеш' in x.value for x in app.markdown)
    assert len(app.get('download_button'))==1


def test_pause_resume_ui_and_no_repeat(tmp_path,monkeypatch):
    from coursework.experiments import run_method
    raw=execute(tmp_path,limit=2,progress=lambda _:None)
    cfg=ProviderConfig(api_key='test-only-key',free_tier_confirmed=True,max_requests=1,delay=1)
    answer={'topic':'Другое','keywords':[],'summary_ru':'','insufficient_data':True}
    session=Mock(post=Mock(return_value=Mock(status_code=200,headers={},json=Mock(return_value={
        'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}))))
    monkeypatch.setattr('coursework.normalize.create_provider',lambda c,ledger,run_id:GeminiProvider(c,ledger,session=session,sleep=lambda _:None,run_id=run_id))
    monkeypatch.setattr(ProviderConfig,'from_env',lambda *a,**k:cfg)
    exp=prepare_experiment(tmp_path,raw/'raw_books.json',2,cfg,purpose='technical')
    run=run_method(tmp_path,exp['experiment_id'],'gemini',config=cfg,api_confirmed=True)
    app=technical(app_for(tmp_path,monkeypatch))
    widget(app.radio,'Метод').set_value('gemini').run()
    page(app,'Нормализация')
    assert not app.exception
    assert any('Обработано 1 из 2' in w.value for w in app.warning)
    widget(app.checkbox,'Только просмотр').uncheck().run()
    widget(app.button,'Продолжить обработку').click().run()
    assert session.post.call_count==1
    assert any('Подтвердите' in w.value for w in app.warning)
    widget(app.number_input,'Лимит запросов на весь запуск').set_value(2)
    widget(app.checkbox,'Подтверждаю продолжение Gemini в пределах доступной бесплатной квоты').check()
    widget(app.button,'Продолжить обработку').click().run()
    assert not app.exception and session.post.call_count==2
    assert read_json(run/'manifest.json')['status']=='completed'
    assert not any('Обработка приостановлена' in w.value for w in app.warning)


def test_human_errors_hide_developer_details():
    assert '.env' not in user_message('missing_or_invalid_api_key')
    assert 'Не удалось обработать' in user_message('invalid_json_response')
    assert 'secret' not in user_message('Traceback secret')
def test_paused_outcomes_include_books_not_yet_attempted(tmp_path):
    from coursework.pipeline import execute
    from coursework.dashboard import load_run
    from coursework.storage import write_json
    from coursework.visuals import chart_data
    directory = execute(tmp_path, limit=10, progress=lambda _: None)
    from coursework.storage import read_json
    run = load_run({'directory': directory, 'manifest': read_json(directory/'manifest.json')})
    run['manifest']['method'] = 'gemini'
    write_json(run['result']/'normalization_stats.json', {'pending_records':8, 'calls':[
        {'row':0,'status':'gemini','error':None,'cached':False},
        {'row':1,'status':'gemini','error':None,'cached':True},
        {'row':2,'status':'paused','error':'run_request_limit'}]})
    frame = chart_data(run)['gemini_outcomes']['data']
    values = dict(frame.values)
    assert values == {'Новые ответы':10.0,'Из кеша':10.0,'Замена правилами':0.0,'Ожидают продолжения':80.0}

