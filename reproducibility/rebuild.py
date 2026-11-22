from pathlib import Path
from hashlib import sha256
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rebuild(destination):
    source=Path(__file__).resolve().parent
    hashes=json.loads((source/'checksums.json').read_text(encoding='utf-8'))
    for name,value in hashes.items():
        if sha256((source/name).read_bytes()).hexdigest()!=value:
            raise ValueError('Набор изменён: '+name)
    rows=pd.read_csv(source/'records.csv',dtype={'id':str})
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    assert set(rows[rows.method=='rules'].id)==set(rows[rows.method=='gemini'].id)
    assert len(rows)==30 and rows.groupby('method').size().eq(15).all()
    destination.mkdir(parents=True,exist_ok=True)
    counts=pd.crosstab(rows.topic,rows.method).reindex(columns=['rules','gemini']).rename(columns={'rules':'Rules','gemini':'Gemini'})
    with plt.rc_context({'font.family':'DejaVu Sans'}):
        ax=counts.plot.barh(color=['#126e75','#4361a8'],figsize=(9,5))
        ax.set(title='Темы книг: Rules и Gemini',xlabel='Количество книг, шт.',ylabel='Тема')
        ax.figure.tight_layout();ax.figure.savefig(destination/'topics.png',dpi=180);plt.close(ax.figure)
    counts.to_csv(destination/'topic_counts.csv',encoding='utf-8-sig')
    result={'books_per_method':15,'same_ids':True,'api_requests':summary['methods']['gemini']['http_requests'],
            'accepted':summary['methods']['gemini']['gemini_processed'],'cache_hits':summary['methods']['gemini']['cache_hits'],
            'api_errors':summary['methods']['gemini']['api_errors'],'limit_pauses':summary['methods']['gemini']['limit_pauses'],
            'tokens':summary['methods']['gemini']['token_usage']['total_tokens']}
    (destination/'reproduced.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Воспроизведение числовых результатов без API')
    parser.add_argument('--out',type=Path,default=Path('outputs/reproduced'))
    print(json.dumps(rebuild(parser.parse_args().out),ensure_ascii=False))
