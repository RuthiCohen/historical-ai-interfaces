from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {
    'exposure': ('appeared', 'deducible', 'did_not_appear'),
    'fidelity': ('supported', 'deducible', 'unsupported'),
}


def human_labels(path, task):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['task'] == task, f'Task mismatch in {path}'
    records = {}
    for case in data['cases']:
        unit_group = case['case_id'] if task == 'exposure' else case['reply_id']
        for item in case['items' if task == 'exposure' else 'claims']:
            if item.get('status') != 'done':
                continue
            unit = item['item_id' if task == 'exposure' else 'claim_id']
            key = (unit_group, unit)
            assert key not in records, f'Duplicate key {key} in {path}'
            assert item['label'] in LABELS[task], (path, key, item['label'])
            records[key] = {'label': item['label'],
                            'source': item.get('availability_source'),
                            'note': item.get('note', ''),
                            'active_seconds': item.get('active_seconds', 0)}
    assert len(records) == data['progress']['done'], f'Count mismatch: {path}'
    return data, records


def llm_labels(path):
    records = {'exposure': {}, 'fidelity': {}}
    with path.open(encoding='utf-8', newline='') as file:
        for r in csv.DictReader(file):
            task = r['task']
            key = (r['record_id'], r['unit_id'])
            assert task in LABELS and r['label'] in LABELS[task]
            assert key not in records[task], f'Duplicate LLM key: {task}/{key}'
            records[task][key] = {'label': r['label'],
                                  'source': r.get('availability_source', '')}
    return records


def collapse(task, label):
    if task == 'exposure':
        return 'available' if label != 'did_not_appear' else 'not_available'
    return 'established' if label != 'unsupported' else 'not_established'


def kappa(a, b, all_labels=None):
    """Unweighted Cohen's kappa: (observed agreement - chance agreement) / (1 - chance agreement)."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    labels = set(a) | set(b) if all_labels is None else set(all_labels)
    p_observed = sum(x == y for x, y in zip(a, b)) / n
    p_chance = sum(ca[k] * cb[k] for k in labels) / (n * n)
    return (p_observed - p_chance) / (1 - p_chance) if p_chance != 1 else None


def compare(x, y, keys, label_map=lambda s:s):
    a = [label_map(x[k]['label']) for k in keys]
    b = [label_map(y[k]['label']) for k in keys]
    agree = sum(i == j for i, j in zip(a, b))
    return {'n':len(keys), 'agree':agree, 'agreement':agree/len(keys),
            'kappa':kappa(a,b)}


def metric_row(task, version, pair, stats):
    return {'task':task,'label_scheme':version,'comparison':pair, **stats}


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_clusters(x, y, keys, transform, seed=2027, trials=3000):
    """Exploratory percentile bootstrap resampling whole conversations/replies.

    These convenience intervals are not a substitute for inference accounting
    for stratification/selection; the paper's main reporting uses exact counts.
    """
    groups=defaultdict(list)
    for key in keys:
        groups[key[0]].append(key)
    ids=sorted(groups)
    r=random.Random(seed)
    agr, kaps=[],[]
    for _ in range(trials):
        resampled=[key for _ in ids for key in groups[r.choice(ids)]]
        a=[transform(x[k]['label']) for k in resampled]
        b=[transform(y[k]['label']) for k in resampled]
        agr.append(sum(v==w for v,w in zip(a,b))/len(a))
        kv=kappa(a,b)
        if kv is not None and math.isfinite(kv):
            kaps.append(kv)
    def pct(t,p):
        if not t: return None
        t=sorted(t); pos=(len(t)-1)*p
        lo=int(math.floor(pos)); hi=int(math.ceil(pos))
        return t[lo]+(pos-lo)*(t[hi]-t[lo])
    return {'agreement_ci95':[pct(agr,.025),pct(agr,.975)],
            'kappa_ci95':[pct(kaps,.025),pct(kaps,.975)],
            'bootstrap_valid_kappa':len(kaps),'bootstrap_trials':trials,
            'clusters':len(ids)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE/'data')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results')
    parser.add_argument('--bootstrap', type=int, default=3000,
                        help='Cluster bootstrap repetitions; use 0 to skip.')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name_patterns={
        'exposure': 'exposure-answers-annotator-*-*.json',
        'fidelity': 'fidelity-answers-annotator-*-*.json'
    }
    llm=llm_labels(args.data_dir/'llm_audit_labels.csv')
    report={}; metrics=[]; confusion=[]; disagreements=[]

    for task,pattern in name_patterns.items():
        paths={}
        for p in args.data_dir.glob(pattern):
            if 'annotator-a-' in p.name: who='A'
            elif 'annotator-b-' in p.name: who='B'
            else: continue
            assert who not in paths, f'Multiple files for {task} {who}'
            paths[who]=p
        assert set(paths)=={'A','B'}, f'Missing annotator for {task}: {paths}'
        meta={}; humans={}
        for who in ('A','B'):
            meta[who], humans[who]=human_labels(paths[who], task)
        assert meta['A']['packet_sha256']==meta['B']['packet_sha256'], 'Different sample packets'
        assert meta['A']['instructions_sha256']==meta['B']['instructions_sha256'], 'Different instructions'

        a,b = humans['A'],humans['B']
        shared=sorted(a.keys() & b.keys())
        assert set(shared) <= set(llm[task]), f'Missing LLM comparisons for {task}'
        models={'A':a, 'B':b, 'LLM':llm[task]}
        info={'n_human_A':len(a),'n_human_B':len(b),
              'n_shared':len(shared),'shared_clusters':len({k[0] for k in shared}),
              'n_LL_full_export':len(llm[task]),
              'n_only_A':len(set(a)-set(b)),'n_only_B':len(set(b)-set(a)),
              'threeway':{},'binary':{}}

        for version,transform in [('threeway',lambda x:x),('binary',lambda x:collapse(task,x))]:
            v={}
            for x,y in [('A','B'),('A','LLM'),('B','LLM')]:
                s=compare(models[x],models[y],shared,transform)
                v[f'{x}_vs_{y}']=s
                metrics.append(metric_row(task,version,f'{x}_vs_{y}',s))
            consensus=[k for k in shared if transform(a[k]['label'])==transform(b[k]['label'])]
            numerator=sum(transform(a[k]['label'])==transform(llm[task][k]['label']) for k in consensus)
            s={'n':len(consensus),'agree':numerator,'agreement':numerator/len(consensus),'kappa':None}
            v['LLM_vs_human_consensus']=s
            metrics.append(metric_row(task,version,'LLM_vs_human_consensus',s))
            v['human_consensus_exclusions']=len(shared)-len(consensus)
            for x in ('A','B','LLM'):
                rate=sum(transform(models[x][k]['label'])==('available' if task=='exposure' else 'established') for k in shared)/len(shared) if version=='binary' else None
                v[f'{x}_positive_rate']=rate
            if args.bootstrap:
                v['A_vs_B_cluster_bootstrap']=bootstrap_clusters(a,b,shared,transform,seed=2027,trials=args.bootstrap)
            info[version]=v
            for u in sorted(set(transform(a[k]['label']) for k in shared)):
                for w in sorted(set(transform(b[k]['label']) for k in shared)):
                    confusion.append({'task':task,'label_scheme':version,'A':u,'B':w,
                        'count':sum(transform(a[k]['label'])==u and transform(b[k]['label'])==w for k in shared)})
            for key in shared:
                if transform(a[key]['label']) != transform(b[key]['label']):
                    disagreements.append({'task':task,'scheme':version,'record_id':key[0],
                        'unit_id':key[1], 'A':a[key]['label'], 'B':b[key]['label'],
                        'LLM':llm[task][key]['label']})

        if task=='exposure':
            rank={'did_not_appear':0,'deducible':1,'appeared':2}
            exact_disagreements=[k for k in shared if a[k]['label'] != b[k]['label']]
            info['adjacent_exact_disagreements']=sum(
                abs(rank[a[k]['label']]-rank[b[k]['label']])==1
                for k in exact_disagreements)
            info['total_exact_disagreements']=len(exact_disagreements)
            source_a=[a[k]['source'] for k in shared]
            source_b=[b[k]['source'] for k in shared]
            exact=sum(x==y for x,y in zip(source_a,source_b))
            info['exposure_source']={'n':len(shared),'agree':exact,
                'agreement':exact/len(shared),'kappa':kappa(source_a,source_b)}
            metrics.append(metric_row(task,'source','A_vs_B',info['exposure_source']))
        info['labels_full_export']=dict(Counter(x['label'] for x in llm[task].values()))
        info['labels_shared']={who:dict(Counter(models[who][k]['label'] for k in shared)) for who in models}
        report[task]=info

    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write_csv(args.output_dir/'agreement_metrics.csv', metrics,
              ['task','label_scheme','comparison','n','agree','agreement','kappa'])
    write_csv(args.output_dir/'confusion_matrices.csv', confusion,
              ['task','label_scheme','A','B','count'])
    write_csv(args.output_dir/'human_disagreements.csv', disagreements,
              ['task','scheme','record_id','unit_id','A','B','LLM'])
    print('Audit validation agreement — raw human JSON matched to LLM export by stable IDs\n')
    for task in ('exposure','fidelity'):
        info=report[task]
        print(f'{task.upper()}: {info["n_shared"]} shared items across {info["shared_clusters"]} cases/replies;'
              f' A {info["n_human_A"]}, B {info["n_human_B"]}, B-only {info["n_only_B"]}')
        for scheme in ('threeway','binary'):
            print(f'  {scheme}:')
            for key,display in [('A_vs_B','Human A vs B'),('A_vs_LLM','LLM vs A'),
                                ('B_vs_LLM','LLM vs B'),('LLM_vs_human_consensus','LLM vs consensus')]:
                s=info[scheme][key]
                kval=f', kappa={s["kappa"]:.3f}' if s['kappa'] is not None else ''
                print(f'    {display}: {s["agree"]}/{s["n"]} ({s["agreement"] * 100:.2f}%){kval}')
        if task=='exposure':
            s=info['exposure_source'];print(f'  evidence-source A vs B: {s["agree"]}/{s["n"]} ({s["agreement"]:.1%}), kappa={s["kappa"]:.3f}')
    print('\nFiles written to',args.output_dir)

if __name__=='__main__':
    main()
