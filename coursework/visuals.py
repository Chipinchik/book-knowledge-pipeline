from io import BytesIO
import pandas as pd
import altair as alt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRIMARY = '#126e75'
SECONDARY = '#4361a8'
FIELD_NAMES = {'title': 'Название', 'category_raw': 'Исходная категория', 'description_raw': 'Описание',
    'upc_raw': 'Идентификатор', 'upc': 'Идентификатор', 'price_raw': 'Цена', 'price_gbp': 'Цена',
    'rating_raw': 'Рейтинг', 'rating': 'Рейтинг', 'availability_raw': 'Наличие', 'source_url': 'Источник',
    'topic': 'Нормализованная тема', 'summary_ru': 'Аннотация'}


def interactive_chart(spec):
    """Stable field names avoid Vega interpreting dots in Russian unit labels."""
    frame, x, y = spec['data'], spec['x'], spec['y']
    scale = alt.Scale(domain=[0,100]) if '%' in y else alt.Scale(zero=True)
    axis = alt.Axis(tickMinStep=1, format='d') if 'шт.' in y else alt.Axis()
    if len(frame.columns)==2:
        data=frame.rename(columns={x:'label',y:'value'})
        chart=alt.Chart(data).mark_bar(color=PRIMARY,cornerRadiusEnd=3).encode(
            y=alt.Y('label:N',title=x,sort=None,axis=alt.Axis(labelLimit=190)),
            x=alt.X('value:Q',title=y,scale=scale,axis=axis),
            tooltip=[alt.Tooltip('label:N',title=x),alt.Tooltip('value:Q',title=y)])
    else:
        data=frame.rename(columns={x:'label'}).melt(id_vars='label',var_name='method',value_name='value')
        chart=alt.Chart(data).mark_bar(cornerRadiusEnd=2).encode(
            y=alt.Y('label:N',title=x,sort=None),yOffset='method:N',
            x=alt.X('value:Q',title=y,scale=scale,axis=axis),
            color=alt.Color('method:N',title='Метод',scale=alt.Scale(domain=['Rules','Gemini'],range=[PRIMARY,SECONDARY]),legend=alt.Legend(orient='top')),
            tooltip=[alt.Tooltip('label:N',title=x),alt.Tooltip('method:N',title='Метод'),alt.Tooltip('value:Q',title=y)])
    return chart.properties(height=max(220,min(380,35*len(frame)))).configure_view(strokeWidth=0)


def chart_data(run):
    records, m = run['records'], run['metrics']
    charts = {}
    def add(key, title, x, y, values):
        charts[key] = {'title': title, 'x': x, 'y': y, 'data': pd.DataFrame(values, columns=[x, y])}
    add('topics', 'Распределение по нормализованным темам', 'Тема', 'Книг, шт.', list(m['topic_counts'].items()))
    prices = pd.Series([r['price_gbp'] for r in records], dtype=float).dropna()
    counts = pd.cut(prices, bins=min(8, max(1, len(prices))), precision=1).value_counts().sort_index() if len(prices) else pd.Series(dtype=int)
    add('prices', 'Распределение учебных цен', 'Цена, GBP (интервал)', 'Книг, шт.', [(str(k), int(v)) for k,v in counts.items()])
    ratings = pd.Series([r['rating'] for r in records], dtype=float).dropna().value_counts().sort_index()
    add('ratings', 'Распределение рейтингов', 'Рейтинг, баллы', 'Книг, шт.', [(str(int(k)), int(v)) for k,v in ratings.items()])
    coverage = m.get('raw_completeness_pct', {})
    add('raw_completeness', 'Полнота исходных данных', 'Поле', 'Заполнено, %', [(FIELD_NAMES.get(k,k),v) for k,v in coverage.items()])
    add('raw_missing', 'Пропуски в исходных данных', 'Поле', 'Пропущено, %', [(FIELD_NAMES.get(k,k),100-v) for k,v in coverage.items()])
    add('duplicates', 'Результат подготовки набора', 'Показатель', 'Записей, шт.', [('Исходные',m['raw_records']),
        ('Нормализованные',m['normalized_records']), ('Дубликаты',m['removed_duplicates']), ('Некорректные',m.get('invalid_raw_records',0))])
    if run['manifest'].get('method') == 'gemini':
        from .storage import read_json
        stats_path = run['result'] / 'normalization_stats.json'
        stats = read_json(stats_path) if stats_path.exists() else {}
        calls = stats.get('calls', [])
        latest = {c['row']:c for c in calls}
        accepted = sum(c.get('status') == 'gemini' and not c.get('error') for c in latest.values())
        cached = sum(c.get('status') == 'gemini' and not c.get('error') and bool(c.get('cached')) for c in latest.values())
        failed = sum(c.get('status') == 'rules_fallback' for c in latest.values())
        pending = stats.get('pending_records', len(latest) - accepted - failed)
        total = accepted + failed + pending
        if total:
            add('gemini_outcomes', 'Результаты Gemini по уникальным записям, включая кеш', 'Результат', 'Доля, %',
                [('Новые ответы', round(100*(accepted-cached)/total,1)), ('Из кеша',round(100*cached/total,1)),
                 ('Замена правилами',round(100*failed/total,1)), ('Ожидают продолжения',round(100*pending/total,1))])
    return charts


def chart_png(spec):
    data, x, y = spec['data'], spec['x'], spec['y']
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        if not data.empty:
            series = [c for c in data.columns if c != x]
            if len(series) == 1:
                ax.barh(data[x].astype(str), data[y].astype(float), color=PRIMARY)
            else:
                data.set_index(x)[series].astype(float).plot.barh(ax=ax, color=[PRIMARY,SECONDARY])
                ax.legend(title='Метод')
        else:
            ax.text(0.5,0.5,'Нет данных', transform=ax.transAxes, ha='center')
        ax.set(title=spec['title'], xlabel=y, ylabel=x)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        fig.tight_layout()
        buffer=BytesIO()
        fig.savefig(buffer, format='png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        return buffer.getvalue()


def comparison_charts(left, right, left_evaluation=None, right_evaluation=None):
    a, b = left['metrics'], right['metrics']
    def paired(title, x, unit, rows):
        return {'title':title,'x':x,'y':unit,'data':pd.DataFrame(rows,columns=[x,'Rules','Gemini'])}
    charts = {'normalization': paired('Полнота нормализованных полей','Поле','Заполнено, %',
        [(FIELD_NAMES.get(k,k),v,b['completeness_pct'].get(k,0)) for k,v in a['completeness_pct'].items()]),
        'time': paired('Измеренное время нормализации (с кешем, если использован)','Этап','Время, с',
            [('Нормализация',a.get('duration_seconds'),b.get('duration_seconds'))]),
        'errors': paired('Ошибки и приостановки','Показатель','Количество, шт.',
            [('Ошибки сервиса',a.get('api_errors',0),b.get('api_errors',0)),
             ('Ошибки ответа',a.get('validation_errors',0),b.get('validation_errors',0)),
             ('Другие ошибки',a.get('other_errors',0),b.get('other_errors',0)),
             ('Паузы по лимиту',a.get('limit_pauses',0),b.get('limit_pauses',0)),
             ('Замена правилами',a.get('llm_fallbacks',0),b.get('llm_fallbacks',0))])}
    if left_evaluation and right_evaluation and left_evaluation.get('topic_labeled_count'):
        charts['manual_quality'] = paired('Классификация по проверенной ручной разметке','Метрика','Значение, 0–1',
            [('Accuracy',left_evaluation['topic_accuracy'],right_evaluation['topic_accuracy']),
             ('Macro-F1',left_evaluation['topic_macro_f1'],right_evaluation['topic_macro_f1'])])
    return charts
