import copy
import json
from hashlib import sha256
from unittest.mock import Mock
import zipfile

import pytest

from coursework.dashboard import list_runs, load_run, compare_runs
from coursework.evaluation import evaluate
from coursework.experiments import prepare_experiment, run_method, load_experiment, exclusive
from coursework.llm_provider import ProviderConfig, GeminiProvider, ProviderError
from coursework.maintenance import archive_development
from coursework.pipeline import execute, ROOT
from coursework.storage import read_json, write_json
from coursework.visuals import chart_data, chart_png, interactive_chart


def test_pair_identity_metadata_idempotence_and_cache(tmp_path, monkeypatch):
    source = execute(tmp_path,limit=2,progress=lambda _:None)
    cfg = ProviderConfig(api_key='test-credential',free_tier_confirmed=True,delay=1,max_output_tokens=1024)
    exp = prepare_experiment(tmp_path,source/'raw_books.json',2,cfg,purpose='technical')
    body = {'topic':'Наука','keywords':['книга'],'summary_ru':'Краткое описание.','insufficient_data':False}
    session = Mock(post=Mock(return_value=Mock(status_code=200,headers={},json=Mock(return_value={
        'choices':[{'finish_reason':'stop','message':{'content':json.dumps(body)}}], 'usage':{'total_tokens':21}}))))
    monkeypatch.setattr('coursework.normalize.create_provider',lambda c,ledger,run_id:GeminiProvider(c,ledger,session=session,sleep=lambda _:None,run_id=run_id))
    a = run_method(tmp_path,exp['experiment_id'],'rules',config=cfg)
    assert read_json(a/'manifest.json')['configuration']=={'provider':'rules','model':None}
    b = run_method(tmp_path,exp['experiment_id'],'gemini',config=cfg,api_confirmed=True)
    assert a != b
    assert (a/'raw_books.json').read_bytes() == (b/'raw_books.json').read_bytes()
    assert session.post.call_count == 2
    assert session.post.call_args.kwargs['json']['max_tokens'] == 1024
    assert session.post.call_args.kwargs['json']['model'] == 'gemini-3.1-flash-lite'
    assert run_method(tmp_path,exp['experiment_id'],'gemini') == b
    assert session.post.call_count == 2
    rows = read_json(b/'gemini'/'knowledge_source.json')
    assert all(r['experiment_id']==exp['experiment_id'] and r['run_id']==b.name and r['processing_status']=='success'
               and not r['is_fallback'] and r['model']==cfg.model for r in rows)
    assert cfg.api_key not in (b/'manifest.json').read_text()
    entries = {e['directory']:e for e in list_runs(tmp_path)}
    assert len(compare_runs(load_run(entries[a]),load_run(entries[b]))) == 2
    with (tmp_path/'experiments'/exp['experiment_id']/'raw_books.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError,match='изменён'):
        load_experiment(tmp_path,exp['experiment_id'])


def test_final_preparation_requires_real_data_and_only_one(tmp_path):
    source = execute(tmp_path,limit=2,progress=lambda _:None)
    raw = read_json(source/'raw_books.json')
    path = tmp_path/'test-only.json'
    # Mark fixture rows as live to test final-run validation.
    rows = [{**raw[0],'source_mode':'live','upc_raw':str(i),'source_url':f'https://books.toscrape.com/{i}'} for i in range(20)]
    write_json(path,rows)
    cfg = ProviderConfig()
    with pytest.raises(ValueError,match='15'):
        prepare_experiment(tmp_path,path,2,cfg)
    with pytest.raises(ValueError,match='1–2'):
        prepare_experiment(tmp_path,path,3,cfg,purpose='technical')
    exp = prepare_experiment(tmp_path,path,15,cfg)
    assert exp['raw_records']==15 and exp['runs']=={}
    with pytest.raises(ValueError,match='уже'):
        prepare_experiment(tmp_path,path,20,cfg)
    assert len(list((tmp_path/'experiments').glob('*/experiment.json')))==1


def test_archive_backup_is_verified_and_excluded(tmp_path):
    old = execute(tmp_path,limit=1,progress=lambda _:None)
    final = execute(tmp_path,raw_path=old/'raw_books.json',purpose='final',progress=lambda _:None)
    (tmp_path/'gemini_cache.sqlite').write_bytes(b'preserve ledger')
    archive = archive_development(tmp_path)
    inventory = read_json(archive/'inventory.json')
    assert inventory['status']=='archived'
    assert not old.exists() and final.exists()
    assert read_json(tmp_path/'latest.json')['run_id']==final.name
    assert len(list_runs(tmp_path))==1
    assert (tmp_path/'gemini_cache.sqlite').read_bytes()==b'preserve ledger'
    with zipfile.ZipFile(archive/'backup.zip') as z:
        for item in inventory['files']:
            data = z.read(item['path'])
            assert sha256(data).hexdigest()==item['sha256']
            assert (archive/'files'/item['path']).read_bytes()==data


def test_archive_preserves_linked_technical_pair(tmp_path):
    source = execute(tmp_path,limit=1,progress=lambda _:None)
    exp = prepare_experiment(tmp_path,source/'raw_books.json',1,ProviderConfig(),purpose='technical')
    run = run_method(tmp_path,exp['experiment_id'],'rules')
    archive_development(tmp_path)
    assert run.exists()
    assert load_experiment(tmp_path,exp['experiment_id'])['runs']['rules']==run.name


def test_unverified_labels_do_not_produce_accuracy():
    records=[{'id':'1','topic':'Наука','summary_ru':''}]
    gold=[{'id':'1','topic_gold':'Наука'}]
    result=evaluate(records,gold)
    assert result['topic_accuracy'] is None and result['topic_macro_f1'] is None
    assert result['topic_unverified_count']==1
    assert evaluate(records,[{**gold[0],'verified':'да'}])['topic_accuracy']==1
    with pytest.raises(ValueError,match='verified'):
        evaluate(records,[{**gold[0],'verified':'maybe'}])


def test_charts_match_saved_records_and_png(tmp_path):
    execute(tmp_path,progress=lambda _:None)
    run=load_run(list_runs(tmp_path)[0])
    charts=chart_data(run)
    assert charts['topics']['data']['Книг, шт.'].sum()==len(run['records'])
    assert charts['prices']['data']['Книг, шт.'].sum()==sum(r['price_gbp'] is not None for r in run['records'])
    assert charts['ratings']['data']['Книг, шт.'].sum()==sum(r['rating'] is not None for r in run['records'])
    assert all(a+b==100 for a,b in zip(charts['raw_completeness']['data'].iloc[:,1],charts['raw_missing']['data'].iloc[:,1]))
    assert chart_png(charts['topics']).startswith(b'\x89PNG\r\n\x1a\n')
    spec = interactive_chart(charts['topics']).to_dict()
    assert spec['encoding']['x']['field']=='value'
    assert sum(r['value'] for rows in spec['datasets'].values() for r in rows)==len(run['records'])


def test_all_environment_parameters_used(monkeypatch):
    values={'MODEL':'gemini-3.1-flash-lite','TIMEOUT':'44','DELAY':'9','MAX_REQUESTS':'16','DAILY_LIMIT':'19',
            'RETRIES':'1','MAX_OUTPUT_TOKENS':'1024','FREE_TIER_CONFIRMED':'true','API_KEY':'test-value'}
    for key,value in values.items():
        monkeypatch.setenv('GEMINI_'+key,value)
    cfg=ProviderConfig.from_env()
    cfg.validate()
    assert (cfg.timeout,cfg.delay,cfg.max_requests,cfg.daily_limit,cfg.retries,cfg.max_output_tokens)==(44,9,16,19,1,1024)
    assert cfg.model==values['MODEL'] and cfg.free_tier_confirmed and cfg.api_key==values['API_KEY']
    assert values['API_KEY'] not in str(cfg.public()) and values['API_KEY'] not in repr(cfg)


def test_plan_validation_and_concurrent_launch_guard(tmp_path):
    with pytest.raises(ProviderError,match='invalid_request_limits'):
        ProviderConfig(max_output_tokens=64).validate_parameters()
    with exclusive(tmp_path/'.lock'):
        with pytest.raises(ValueError,match='выполняется'):
            with exclusive(tmp_path/'.lock'):
                pass
    assert not (tmp_path/'.lock').exists()
