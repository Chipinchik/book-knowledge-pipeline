from dataclasses import replace
from contextlib import closing
from pathlib import Path
import os
import sqlite3
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from coursework.dashboard import list_runs, load_run, filter_records, compare_runs, manual_metrics
from coursework.experiments import list_experiments, load_experiment, prepare_experiment, run_method
from coursework.llm_provider import ProviderConfig, ProviderError
from coursework.pipeline import ROOT, execute, process_run
from coursework.scraper import allowed_url
from coursework.storage import read_json
from coursework.status import user_message, event_metrics
from coursework.visuals import chart_data, chart_png, comparison_charts, interactive_chart

OUTPUT = Path(os.getenv('BOOK_PIPELINE_OUTPUT', str(ROOT / 'outputs')))
PAGES = ['Главная', 'Сбор данных', 'Нормализация', 'Каталог книг', 'Аналитика', 'Сравнение методов', 'Экспорт и эксперименты', 'Помощь']
STATUSES = {'completed':'Завершён', 'completed_with_errors':'Завершён с ошибками', 'collected':'Снимок собран',
    'processing_paused':'Приостановлен', 'processing_interrupted':'Прерван', 'processing_failed':'Ошибка обработки',
    'collection_failed':'Ошибка сбора', 'interrupted':'Сбор прерван', 'processing':'Выполняется', 'created':'Создан'}
METHODS = {'rules':'Rules', 'gemini':'Gemini', 'rules_fallback':'Rules после ошибки Gemini'}
def error(exc):
    code = str(exc) if isinstance(exc, (ValueError, ProviderError)) else type(exc).__name__
    st.error(user_message(code))


def date_label(value):
    try:
        return datetime.fromisoformat(value).astimezone(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y %H:%M:%S')
    except (ValueError, TypeError):
        return 'Дата не указана'


def experiment_label(value):
    return f"{value.get('raw_records',0)} книг · {date_label(value.get('created_at'))}"


def new_experiment():
    st.session_state['creating'] = True
    st.session_state['view_only'] = False
    st.session_state['page'] = 'Сбор данных'


def progress_callback(total):
    bar=st.progress(0.0, text='Подготовка обработки…')
    status=st.empty()
    def update(message):
        match=re.search(r'(?:Обработка|Книга)[: ]+(\d+)/(\d+)',message)
        if match:
            done,n=map(int,match.groups())
            bar.progress(min(1.0,done/max(n,1)), text=f'Обработано {done} из {n} записей')
        elif 'Страница' in message or 'Источник' in message:
            status.info('Получаем данные из источника…')
    return update


def progress_summary(manifest, metrics=None):
    m=metrics or manifest.get('metrics',{})
    total=manifest.get('raw_records',0)
    pending=m.get('pending_records',max(0,total-m.get('normalized_records',0)))
    done=max(0,total-pending)
    st.progress(done/max(1,total),text=f'Обработано {done} из {total} записей')
    cols=st.columns(3)
    cols[0].metric('Успешные записи',m.get('normalized_records',0)-m.get('llm_fallbacks',0))
    cols[1].metric('События ошибок',m.get('processing_errors',0),help='Ошибки сервиса, проверки ответа и другие ошибки. Приостановки по локальному лимиту считаются отдельно.')
    cols[2].metric('Осталось записей',pending)
    if m.get('llm_fallbacks'):
        st.warning(f"Для {m['llm_fallbacks']} записей использованы правила вместо Gemini. Они не считаются успешными ответами модели.")
    if manifest.get('status') in ('processing_paused','processing_failed','processing_interrupted','processing'):
        st.warning(f'Обработка приостановлена. Обработано {done} из {total} записей. '+user_message(manifest.get('pause_reason','processing_interrupted')))
    elif manifest.get('status') in ('completed','completed_with_errors'):
        st.success('Обработка завершена. Сохранённые результаты готовы к просмотру.')


def technical_info(experiment=None,run=None):
    with st.expander('Техническая информация'):
        st.caption('Идентификаторы, параметры и контрольные суммы нужны для воспроизведения исследования. Просмотр не изменяет файлы.')
        if experiment:
            st.json(experiment)
        if run:
            st.json(run['manifest'])
            st.write('Уточнённая классификация событий', {k:run['metrics'].get(k) for k in ('api_errors','validation_errors','limit_pauses','safety_pauses','other_errors','original_error_events')})
            st.caption('Исходные metrics.json и журнал не перезаписаны. Счётчики интерфейса рассчитываются из сохранённых событий; число событий не равно числу HTTP-ошибок внутри повторных попыток.')


def go(page):
    st.session_state['page'] = page


def focus(identifier, purpose):
    st.session_state['pending_selection'] = (identifier, purpose != 'final')
    if identifier.startswith('exp-'):
        st.session_state['creating'] = False
    st.rerun()


@st.cache_data(show_spinner=False)
def png(spec):
    return chart_png(spec)


def show_chart(key, spec):
    st.markdown('#### ' + spec['title'])
    frame, x, y = spec['data'], spec['x'], spec['y']
    if frame.empty:
        st.caption('Нет данных для построения диаграммы.')
        return
    chart = interactive_chart(spec)
    st.altair_chart(chart, width='stretch')
    st.download_button('Скачать PNG', png(spec), key+'.png', 'image/png', key='png_'+key)


def card(label, value, help_text=None):
    with st.container(border=True):
        st.metric(label, value, help=help_text)


def overview(context, run, experiment, view_only):
    st.write('Собирайте сведения о книгах, приводите данные к единому виду и сравнивайте результаты обработки.')
    if not context:
        st.info('Начните со сбора данных. После обработки здесь появятся каталог и результаты анализа.')
        st.button('Начать работу',on_click=new_experiment,type='primary')
        return
    st.subheader('Сохранённый эксперимент' if experiment else 'Сохранённый набор книг')
    st.caption('Просмотр ранее полученных результатов. Новые запросы к Gemini не выполняются.')
    if run:
        m=run['metrics']
        cols=st.columns(3)
        indicators=[('Собрано книг','raw_records'),('Нормализовано','normalized_records'),('Структурированных фактов','exported_facts'),
            ('Пропущенных значений','missing_values'),('Удалено дубликатов','removed_duplicates'),('Требуют проверки','rows_requiring_review')]
        for i,(label,key) in enumerate(indicators):
            with cols[i%3]:
                card(label,m.get(key,'—'))
        st.caption('Пропуски включают аннотацию: Rules не создаёт её автоматически. Цена указана в GBP, рейтинг — от 1 до 5.')
        st.write(f"Метод: **{METHODS.get(run['manifest'].get('method'),'Сбор данных')}** · {STATUSES.get(run['manifest'].get('status'),'Результаты сохранены')}")
        if run['manifest'].get('status') not in ('completed','completed_with_errors'):
            progress_summary(run['manifest'],m)
        st.dataframe(pd.DataFrame(run['records'][:3],columns=['title','topic','price_gbp']).rename(columns={'title':'Последние результаты','topic':'Тема','price_gbp':'Цена, GBP'}),hide_index=True,width='stretch')
    else:
        st.info('Исходные книги сохранены. Следующий шаг — выбрать метод и выполнить нормализацию.')
    st.markdown('#### Gemini')
    st.caption('Соединение не проверялось: для просмотра оно не требуется. Подключение используется только при подтверждённом запуске обработки.')
    if experiment and 'gemini' in experiment['runs']:
        st.write('Сохранённые результаты Gemini доступны: выберите метод в боковой панели.')
    cols=st.columns(3)
    cols[0].button('Открыть каталог',on_click=go,args=('Каталог книг',),width='stretch')
    cols[1].button('Сравнить методы',on_click=go,args=('Сравнение методов',),width='stretch')
    cols[2].button('Экспортировать',on_click=go,args=('Экспорт и эксперименты',),width='stretch')
    technical_info(experiment,run)


def collect(view_only, entries):
    st.write('Создайте исходный снимок. Сбор не вызывает Gemini и не запускает сравнение автоматически.')
    if view_only:
        st.info('Для сбора отключите «Только просмотр» в боковой панели. Сохранённые данные доступны без сети.')
    source=st.selectbox('Источник', ['live','demo'], format_func=lambda v:'Books to Scrape — реальные страницы' if v=='live' else 'Демонстрационные HTML — без сети')
    with st.form('collect'):
        c1,c2=st.columns(2)
        count=c1.number_input('Количество книг',1,30,20 if source=='live' else 10)
        with c2.expander('Параметры сбора'):
            delay=st.number_input('Пауза между запросами, с',1.0,60.0,1.5,0.5,help='Пауза снижает нагрузку на сайт.')
        submit=st.form_submit_button('Собрать данные', disabled=view_only, type='primary')
    st.caption('Источник — учебный сайт Books to Scrape. Исходные страницы и сведения о книгах сохраняются автоматически.')
    if submit:
        progress=st.empty()
        try:
            with st.spinner('Сохраняем страницы и исходные записи…'):
                directory=execute(OUTPUT,source=source,limit=count,delay=delay,scrape_only=True,progress=progress.text)
            focus(directory.name,'technical')
        except Exception as exc:
            error(exc)
    snapshots=[e for e in entries if e['manifest'].get('raw_sha256')]
    if snapshots:
        entry=st.selectbox('Просмотр исходного снимка',snapshots,format_func=lambda e:f"Набор {entries.index(e)+1} · "+experiment_label(e['manifest']))
        raw=read_json(entry['directory']/'raw_books.json')
        st.subheader('Исходные книги')
        st.dataframe(pd.DataFrame(raw,columns=['title','category_raw','price_raw','rating_raw']).rename(columns={'title':'Название','category_raw':'Категория','price_raw':'Цена, GBP','rating_raw':'Рейтинг источника'}),hide_index=True,width='stretch',height=300)
        st.button('Далее: нормализация',on_click=go,args=('Нормализация',),type='primary')
        with st.expander('Подробности исходных данных'):
            st.json(raw)
        st.download_button('Исходный JSON',(entry['directory']/'raw_books.json').read_bytes(),'raw_books.json','application/json')


def normalize_page(experiment, view_only, entries, method):
    st.caption('Rules приводит поля к единому виду. Gemini также анализирует описание, выделяет тему, ключевые слова и аннотацию.')
    existing=experiment['runs'].get(method) if experiment else None
    if existing:
        directory=OUTPUT/'runs'/existing
        manifest=read_json(directory/'manifest.json')
        entry=next(e for e in entries if e['directory'].name==existing)
        try:
            saved=load_run(entry)
            m=saved['metrics']
        except (OSError,ValueError,KeyError):
            saved=None
            stats=directory/method/'normalization_stats.json'
            data=read_json(stats) if stats.exists() else {}
            checkpoint=directory/method/'processing_checkpoint.json'
            records=read_json(checkpoint).get('records',[]) if checkpoint.exists() else []
            m={**manifest.get('metrics',{}),**event_metrics(data),'pending_records':data.get('pending_records',manifest.get('raw_records',0)),
               'normalized_records':len(records),'llm_fallbacks':sum(r.get('enrichment_method')=='rules_fallback' for r in records)}
        progress_summary(manifest,m)
        if manifest['status'] in ('completed','completed_with_errors'):
            st.info('Этот метод уже выполнен. Повторная обработка не нужна.')
            st.button('Посмотреть результаты',on_click=go,args=('Каталог книг',),type='primary')
            technical_info(experiment,saved)
            return
        if view_only:
            st.info('Чтобы продолжить, отключите «Только просмотр». Уже полученные ответы будут сохранены.')
            technical_info(experiment,saved)
            return
        if manifest['status']=='processing':
            st.warning('Запуск может ещё выполняться. Не запускайте второй процесс. При аварийном завершении обратитесь к администратору.')
            return
        if method=='gemini' and manifest['status'] in ('processing_paused','processing_interrupted','processing_failed'):
            with st.form('resume'):
                with st.expander('Настройки продолжения'):
                    cap=st.number_input('Лимит запросов на весь запуск',1,100,manifest['configuration']['max_requests'],help='Включает уже выполненные попытки. Увеличивайте только в пределах доступной бесплатной квоты.')
                    retry=st.checkbox('Повторить ранее отклонённые запросы',help='Только после устранения причины отказа сервиса. Успешные ответы не отправляются повторно.')
                consent=st.checkbox('Подтверждаю продолжение Gemini в пределах доступной бесплатной квоты')
                submit=st.form_submit_button('Продолжить обработку',type='primary')
            if submit:
                if not consent:
                    st.warning('Подтвердите обращение к Gemini перед продолжением.')
                    return
                try:
                    base=ProviderConfig.from_env()
                    frozen=manifest['configuration']
                    cfg=replace(base,**{k:frozen[k] for k in ('model','timeout','delay','daily_limit','retries','max_output_tokens')},max_requests=cap,retry_failed=retry,free_tier_confirmed=True)
                    process_run(directory,enrich=method,config=cfg,resume=True,api_confirmed=True,progress=progress_callback(experiment['raw_records']))
                    st.rerun()
                except Exception as exc:
                    error(exc)
        elif method=='rules' and manifest['status'] in ('processing_interrupted','processing_failed'):
            if st.button('Продолжить обработку'):
                try:
                    process_run(directory,enrich='rules',resume=True,progress=progress_callback(experiment['raw_records']))
                    st.rerun()
                except Exception as exc:
                    error(exc)
        technical_info(experiment,saved)
        return
    if view_only:
        st.info('Обработка ещё не запускалась. Чтобы начать, отключите «Только просмотр». Просмотр сохранённых данных работает без подключения.')
        return
    try:
        base=ProviderConfig.from_env()
    except ProviderError as exc:
        error(exc)
        return
    if experiment:
        st.subheader(f"{METHODS[method]} · {experiment['raw_records']} книг")
        consent=False
        if method=='gemini':
            st.info('Для обработки требуется соединение с Google Gemini и доступная бесплатная квота. Подготовка и просмотр не расходуют запросы.')
            st.write('Подключение настроено, но не проверено.' if base.api_key else 'Подключение не настроено. Откройте руководство по настройке в разделе «Помощь».')
            st.link_button('Проверить доступную квоту','https://aistudio.google.com/rate-limit')
            consent=st.checkbox('Подтверждаю реальные запросы Gemini в пределах доступной бесплатной квоты')
        if st.button('Запустить '+METHODS[method],type='primary',disabled=method=='gemini' and not consent):
            try:
                with st.spinner('Обрабатываем книги и сохраняем результаты…'):
                    run_method(OUTPUT,experiment['experiment_id'],method,api_confirmed=consent,
                        config=replace(base,free_tier_confirmed=consent),progress=progress_callback(experiment['raw_records']))
                st.rerun()
            except Exception as exc:
                error(exc)
        technical_info(experiment)
        return
    snapshots=[e for e in entries if e['manifest'].get('raw_sha256')]
    if not snapshots:
        st.info('Сначала соберите книги в разделе «Сбор данных».')
        st.button('Перейти к сбору данных',on_click=go,args=('Сбор данных',))
        return
    st.subheader('Подготовка нового эксперимента')
    source=st.selectbox('Исходный снимок',snapshots,format_func=lambda e:f"Набор {entries.index(e)+1} · "+experiment_label(e['manifest']))
    rows=read_json(source['directory']/'raw_books.json')
    if not rows:
        st.warning('Выбранный набор пуст. Соберите данные или выберите другой набор.')
        return
    with st.expander('Параметры исследования'):
        kind=st.radio('Назначение',['final','technical'],format_func=lambda v:'Сравнение на 15–30 книгах' if v=='final' else 'Короткая проверка на 1–2 книгах',horizontal=True)
    minimum,maximum=(15,min(30,len(rows))) if kind=='final' else (1,min(2,len(rows)))
    if maximum<minimum:
        st.info('Для сравнения соберите не менее 15 реальных книг. Для меньшего набора выберите короткую проверку в параметрах исследования.')
        return
    with st.form('prepare'):
        count=st.number_input('Количество книг для обработки',minimum,maximum,min(20,maximum),help='Оба метода обработают одинаковые первые записи выбранного набора.')
        with st.expander('Настройки подключения и ограничения'):
            model=st.text_input('Модель Gemini',base.model,help='Проверьте доступность модели в своём проекте Google. После подготовки эксперимента модель фиксируется.')
            c1,c2=st.columns(2)
            cap=c1.number_input('Лимит запросов на весь запуск',1,100,base.max_requests,help='Считает также повторные попытки. Достижение лимита приостанавливает обработку без потери результатов.')
            daily=c2.number_input('Локальный суточный лимит',1,1000,base.daily_limit,help='Не учитывает запросы из других программ. Фактическую квоту показывает Google AI Studio.')
            delay=c1.number_input('Пауза между запросами, с',1.0,300.0,float(base.delay))
            retries=c2.number_input('Повторы при временном отказе сервиса',0,2,base.retries)
            tokens=c1.number_input('Максимум выходных токенов',128,2048,base.max_output_tokens,128,help='Токены — единицы объёма текста. Малый бюджет может обрезать ответ; превышение бюджета не исправляется автоматически.')
            timeout=c2.number_input('Время ожидания ответа, с',1.0,600.0,float(base.timeout))
        st.caption('Подготовка сохраняет отдельный набор для двух методов. Ранее выполненные эксперименты не меняются. Запросы Gemini начнутся только после отдельного подтверждения.')
        submit=st.form_submit_button('Подготовить общий эксперимент',type='primary')
    if submit:
        try:
            cfg=replace(base,model=model,max_output_tokens=tokens,max_requests=cap,daily_limit=daily,delay=delay,retries=retries,timeout=timeout)
            value=prepare_experiment(OUTPUT,source['directory']/'raw_books.json',count,cfg,kind,allow_new=bool(st.session_state.get('creating')))
            focus(value['experiment_id'],kind)
        except Exception as exc:
            error(exc)


def catalog(run):
    records=run['records']
    search=st.text_input('Поиск по названию',placeholder='Введите часть названия')
    with st.expander('Фильтры и сортировка',expanded=True):
        c1,c2=st.columns(2)
        topics=sorted({r['topic'] for r in records})
        selected=c1.multiselect('Темы',topics,default=topics)
        methods=sorted({r['enrichment_method'] for r in records})
        selected_methods=c2.multiselect('Метод записи',methods,default=methods,format_func=lambda s:METHODS.get(s,s))
        c1,c2=st.columns(2)
        ratings=sorted({r['rating'] for r in records if r['rating'] is not None})
        chosen=c1.multiselect('Рейтинг, баллы',ratings,default=ratings)
        prices=[r['price_gbp'] for r in records if r['price_gbp'] is not None]
        price_range=c2.slider('Цена, GBP',float(min(prices)),float(max(prices)),(float(min(prices)),float(max(prices)))) if prices and min(prices)<max(prices) else None
        missing=st.checkbox('Включать записи с отсутствующей ценой или рейтингом',True)
    filtered=filter_records(records,search,selected,chosen+([None] if missing else []),selected_methods,price_range,missing)
    st.caption(f'Найдено {len(filtered)} из {len(records)} · сортировка нажатием на заголовок столбца')
    columns=['title','topic','price_gbp','rating','enrichment_method']
    table=pd.DataFrame(filtered,columns=columns)
    table['enrichment_method']=table['enrichment_method'].map(METHODS)
    st.dataframe(table,hide_index=True,width='stretch',height=320,
        column_config={'title':st.column_config.TextColumn('Название',width='large'),'topic':'Тема','price_gbp':st.column_config.NumberColumn('Цена, GBP',format='%.2f'),
                       'rating':'Рейтинг','enrichment_method':'Метод'})
    if filtered:
        book=st.selectbox('Подробная карточка',filtered,format_func=lambda r:r['title'])
        left,right=st.columns(2)
        with left:
            st.markdown('#### Исходное описание')
            st.text(book['description_raw'] or 'Описание отсутствует')
            st.caption('Исходная категория: '+book['category_raw'])
            if allowed_url(book['source_url']):
                st.link_button('Страница источника',book['source_url'])
        with right:
            st.markdown('#### Результат нормализации')
            st.write('Тема: '+book['topic'])
            st.text(book['summary_ru'] or 'Аннотация не создавалась')
            st.write('Ключевые слова: '+(', '.join(book['keywords']) or 'нет'))
            st.caption('Модель: '+str(book.get('model') or 'не используется'))
            if book['quality_flags']:
                st.warning('Некоторые сведения отсутствуют или требуют проверки. Сверьте результат с исходным описанием.')
        with st.expander('Техническая информация о книге'):
            st.json(book)
            raw_path=run['directory']/'raw_books.json'
            stats_path=run['result']/'normalization_stats.json'
            if raw_path.exists() and stats_path.exists():
                raw=read_json(raw_path)
                index=next((i for i,r in enumerate(raw) if isinstance(r,dict) and r.get('source_url')==book['source_url']),None)
                if index is not None:
                    st.json(raw[index])
                    for event in read_json(stats_path).get('calls',[]):
                        if event['row']==index:
                            st.json(event)


def compare(experiment, entries):
    if not experiment or not all(k in experiment['runs'] for k in ('rules','gemini')):
        st.info('Для сравнения нужны два завершённых метода одного общего эксперимента. Нет результатов Gemini — нет сравнительных показателей.')
        return
    lookup={e['directory'].name:e for e in entries}
    try:
        a,b=[load_run(lookup[experiment['runs'][k]]) for k in ('rules','gemini')]
        differences=compare_runs(a,b)
        gold=a['result']/'gold_labels.csv'
        ea,eb=manual_metrics(a,gold),manual_metrics(b,gold)
    except (ValueError,OSError,KeyError) as exc:
        error(exc)
        return
    st.success('Снимок и идентификаторы книг совпадают. Показаны только методы выбранного эксперимента.')
    st.caption('Сравнение выполнено по одинаковым книгам и исходным данным.')
    keys={'normalized_records':'Нормализованных записей','gemini_processed':'Принятых ответов Gemini','accepted_new_responses':'Принятых новых ответов','llm_fallbacks':'Замена ответа правилами',
          'api_errors':'Ошибки API','validation_errors':'Ошибки валидации','other_errors':'Другие ошибки','limit_pauses':'Приостановки по локальному лимиту','safety_pauses':'Защитные приостановки','http_requests':'HTTP-попыток','cache_hits':'Ответов из кеша','duration_seconds':'Время нормализации, с'}
    table=pd.DataFrame({'Показатель':list(keys.values()),'Rules':[a['metrics'].get(k) for k in keys],'Gemini':[b['metrics'].get(k) for k in keys]})
    st.dataframe(table,hide_index=True,width='stretch')
    st.caption('Время включает обработку с кешем. При кешировании оно не равно времени нового инференса. Токены отсутствуют, если API их не сообщил.')
    st.caption('Приостановки по локальному лимиту не являются ошибками API. Счётчики уточнены по журналу; исходные файлы сохранены без изменений.')
    st.metric('Токены новых запросов Gemini',b['metrics'].get('token_usage',{}).get('total_tokens','Нет сведений'),help='Токены — единицы объёма текста. Ответы из кеша не требуют нового запроса.')
    for key,spec in comparison_charts(a,b,ea,eb).items():
        show_chart('comparison_'+key,spec)
    if not ea['topic_labeled_count']:
        st.info('Accuracy и Macro-F1 не измерены: сначала нужна проверенная ручная оценка тем книг. Инструкция находится в разделе «Помощь».')
    st.write('Проверено тем: ',ea['topic_labeled_count'])
    st.caption('Метрики классификации оценивают весь конвейер, включая явно отмеченные fallback. Это не точность только успешных ответов Gemini; доля fallback приведена отдельно.')
    with st.expander('Подробности оценки качества'):
        st.json({'Rules':ea,'Gemini':eb})
    display=differences.rename(columns={'title':'Название','rules_topic':'Тема Rules','llm_topic':'Тема Gemini','llm_status':'Статус Gemini','different':'Есть различия'})
    display['Статус Gemini']=display['Статус Gemini'].map(METHODS)
    st.dataframe(display[['Название','Тема Rules','Тема Gemini','Статус Gemini','Есть различия']],hide_index=True,width='stretch')
    technical_info(experiment,b)
    st.download_button('Скачать полное сравнение CSV',differences.to_csv(index=False).encode('utf-8-sig'),'comparison.csv','text/csv')


def exports(run,experiment,entries):
    st.write('Скачайте данные выбранного эксперимента. CSV удобно открывать в таблицах, JSON — использовать в программах, SQLite — как локальную базу данных.')
    related=[e for e in entries if experiment and e['manifest'].get('experiment_id')==experiment['experiment_id']]
    if related:
        st.dataframe(pd.DataFrame([{'Метод':METHODS.get(e['manifest'].get('method'),'Сбор'),'Статус':STATUSES.get(e['manifest']['status'],e['manifest']['status']),
            'Создан':date_label(e['manifest'].get('created_at')),'Модель':e['manifest'].get('configuration',{}).get('model') or 'Не используется'} for e in related]),hide_index=True,width='stretch')
    if run:
        primary={'knowledge_source.csv':'Каталог CSV','knowledge_source.json':'Каталог JSON','knowledge_source.sqlite':'База SQLite','facts_triples.csv':'Факты CSV'}
        cols=st.columns(2)
        for i,(name,label) in enumerate(primary.items()):
            path=run['result']/name
            if path.exists():
                cols[i%2].download_button(label,path.read_bytes(),name,key='export_'+name,width='stretch')
        with st.expander('Материалы для оценки качества'):
            for name,label in {'gold_labels.csv':'Эталон тем','summary_reviews.csv':'Оценка аннотаций','fact_reviews.csv':'Оценка фактов','evaluation.json':'Ручные метрики'}.items():
                path=run['result']/name
                if path.exists():
                    st.download_button(label,path.read_bytes(),name,key='export_'+name)
        with st.expander('Техническая информация и журналы'):
            for name in ('metrics.json','schema.json','normalization_stats.json'):
                path=run['result']/name
                if path.exists():
                    st.download_button('Исходный '+name,path.read_bytes(),name,key='export_'+name)
            st.download_button('Уточнённая статистика событий',json.dumps(run['metrics'],ensure_ascii=False,indent=2).encode('utf-8'),'event_metrics.json','application/json')
            st.caption('metrics.json — оригинал. event_metrics.json — отдельное представление с классификацией ошибок и приостановок; оригинальные результаты не изменяются.')
        with st.expander('Просмотр SQLite и происхождения'):
            database=run['result']/'knowledge_source.sqlite'
            if database.exists():
                with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as db:
                    st.write('Таблицы: books и facts. Просмотр первых 30 фактов.')
                    st.dataframe(pd.read_sql_query('SELECT subject,predicate,object,source_url FROM facts LIMIT 30',db),hide_index=True,width='stretch')
            st.json(run['manifest'])
        st.markdown('#### Диаграммы для пояснительной записки')
        for key,spec in chart_data(run).items():
            if not spec['data'].empty:
                st.download_button(spec['title']+' · PNG',png(spec),key+'.png','image/png',key='export_png_'+key)
    else:
        st.info('Выберите выполненный метод, чтобы скачать его результаты.')


def main():
    st.set_page_config(page_title='Каталог знаний',layout='wide',initial_sidebar_state='expanded')
    st.markdown('<style>.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1480px} h1{font-size:2rem!important} [data-testid="stMetricValue"]{font-size:1.8rem}</style>',unsafe_allow_html=True)
    entries=list_runs(OUTPUT)
    experiments=list_experiments(OUTPUT)
    pending=st.session_state.pop('pending_selection',None)
    if pending:
        st.session_state['show_other']=pending[1]
        st.session_state['selected_id']=pending[0]
    with st.sidebar:
        st.subheader('Каталог знаний')
        st.caption('Книги · данные · сравнение методов')
        page=st.radio('Раздел',PAGES,key='page',label_visibility='collapsed')
        st.divider()
        with st.expander('Другие сохранённые наборы'):
            show_other=st.checkbox('Показать короткие проверки и исходные наборы',key='show_other')
        visible=[e for e in experiments if e['purpose']=='final' or show_other]
        options={e['experiment_id']:{'experiment':e,'label':f'Сравнение {i+1} · '+experiment_label(e)} for i,e in enumerate(visible)}
        if show_other:
            for i,entry in enumerate(entries):
                m=entry['manifest']
                if not str(m.get('experiment_id','')).startswith('exp-'):
                    options[entry['directory'].name]={'entry':entry,'label':f'Набор {i+1} · '+experiment_label(m)}
        if st.session_state.get('selected_id') not in options:
            st.session_state.pop('selected_id',None)
        selected=st.selectbox('Эксперимент',list(options),format_func=lambda k:options[k]['label'],key='selected_id') if options else None
        context=options.get(selected)
        experiment=None
        if context and 'experiment' in context:
            try:
                experiment=load_experiment(OUTPUT,selected)
            except (ValueError,OSError) as exc:
                error(exc)
        method=st.radio('Метод',['rules','gemini'],format_func=lambda s:METHODS[s],horizontal=True,key='method') if experiment else 'rules'
        view_only=st.checkbox('Только просмотр',value=True,key='view_only',help='Сохранённые результаты доступны без ключа и интернета. Обработка и сбор отключены.')
        st.button('Новый эксперимент',on_click=new_experiment,width='stretch',help='Создать отдельный опыт. Сохранённые результаты останутся неизменными.')
        if st.session_state.get('creating') and st.button('Отменить подготовку'):
            st.session_state['creating']=False
            st.rerun()
        st.caption('Источник: Books to Scrape. Учебные цены и рейтинги не описывают книжный рынок.')
    run=None
    if context:
        entry=context.get('entry')
        if experiment and method in experiment['runs']:
            entry=next((e for e in entries if e['directory'].name==experiment['runs'][method]),None)
        if entry and entry['manifest'].get('result_dir'):
            try:
                run=load_run(entry)
            except (ValueError,OSError,KeyError) as exc:
                error(exc)
    st.title('Каталог знаний' if page=='Главная' else page)
    if context:
        st.caption(context['label']+' · '+METHODS[method])
    if run and run['metrics']['source_mode']=='demo':
        st.warning('Демонстрационные синтетические данные. Они не получены с реального сайта.')
    if page=='Главная':
        overview(context,run,experiment,view_only)
    elif page=='Сбор данных':
        if st.session_state.get('creating'):
            st.info('Новый эксперимент: соберите книги или перейдите к нормализации уже сохранённого набора.')
            st.button('Использовать сохранённые книги',on_click=go,args=('Нормализация',))
        collect(view_only,entries)
    elif page=='Нормализация':
        normalize_page(None if st.session_state.get('creating') else experiment,view_only,entries,method)
    elif page=='Каталог книг':
        if run:
            catalog(run)
        else:
            st.info('Каталог появится после нормализации. Исходные книги доступны в разделе «Сбор данных».')
    elif page=='Аналитика':
        if run:
            specs=chart_data(run)
            for i in range(0,len(specs),2):
                for col,(key,spec) in zip(st.columns(2),list(specs.items())[i:i+2]):
                    with col:
                        show_chart(key,spec)
        else:
            st.info('Сначала выберите сохранённый результат обработки.')
    elif page=='Сравнение методов':
        compare(experiment,entries)
    elif page=='Помощь':
        st.markdown((ROOT/'USER_GUIDE.md').read_text(encoding='utf-8'))
        with st.expander('Подключение и расширенные настройки'):
            st.markdown('Для нового обращения к Gemini администратор настраивает ключ и модель в локальном файле `.env` по `.env.example`. Не передавайте ключ другим людям. Модель, паузы и лимиты можно настроить при подготовке нового эксперимента. Для уже сохранённых результатов подключение не нужно.')
        st.download_button('Скачать руководство',(ROOT/'USER_GUIDE.md').read_bytes(),'USER_GUIDE.md','text/markdown')
    else:
        exports(run,experiment,entries)


if __name__=='__main__':
    main()
