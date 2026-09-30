"""Summarize fixed validation comparisons. Run on the experiment server."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();root=Path(a.run_dir)
arrays={f'mixformer_{L}':np.load(root/'history_groups'/f'val_seq{L}.npz') for L in [50,100,200]}
arrays['target_attention_50']=np.load(root/'baseline_history_groups'/'val_seq50.npz')
ref=arrays['mixformer_50'];lengths=ref['lengths'];n=len(lengths)
assert n==6034
for v in arrays.values():
 for field in ['user_ids','lengths','candidates']:assert np.array_equal(v[field],ref[field])
 assert np.isfinite(v['scores']).all()
 assert np.array_equal(v['ranks'],1+(v['scores'][:,1:]>=v['scores'][:,:1]).sum(1))
buckets={'<=20':lengths<=20,'21-50':(lengths>20)&(lengths<=50),'51-100':(lengths>50)&(lengths<=100),'101-200':(lengths>100)&(lengths<=200),'>200':lengths>200,'>100':lengths>100,'overall':lengths>0}
assert sum(int(buckets[k].sum()) for k in ['<=20','21-50','51-100','101-200','>200'])==n
scores={name:{'hr@10':(v['ranks']<=10).astype(float),'ndcg@10':(v['ranks']<=10)/np.log2(v['ranks']+1)} for name,v in arrays.items()}
result={'n':n,'protocol_identical':True,'metrics':{},'paired_differences':{},'ci_note':'2000 paired-user bootstrap resamples, percentile 95% CI, seed 20260916; conditional on fixed trained checkpoints, not training-seed uncertainty; unadjusted exploratory intervals'}
for name,m in scores.items():
 result['metrics'][name]={k:{'n':int(mask.sum()),**{metric:float(v[mask].mean()) for metric,v in m.items()}} for k,mask in buckets.items()}
for name in ['mixformer_100','mixformer_200','target_attention_50']:
 result['paired_differences'][name+' minus mixformer_50']={}
 for bucket,mask in buckets.items():
  r={};rng=np.random.default_rng(20260916)
  for metric in ['hr@10','ndcg@10']:
   d=(scores[name][metric]-scores['mixformer_50'][metric])[mask]
   # Small chunks avoid large temporary allocations.
   means=np.concatenate([d[rng.integers(len(d),size=(100,len(d)))].mean(1) for _ in range(20)])
   r[metric]={'difference':float(d.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n':len(d)}
  result['paired_differences'][name+' minus mixformer_50'][bucket]=r
training=json.loads((root/'target_attention'/'metrics.json').read_text())
h=training['history'];assert len(h)==24
best=max(h,key=lambda x:x['ndcg@10']);first12=max(h[:12],key=lambda x:x['ndcg@10'])
assert np.isclose(best['ndcg@10'],result['metrics']['target_attention_50']['overall']['ndcg@10'],atol=1e-6)
result['baseline_training']={'epochs':len(h),'best_epoch':best['epoch'],'best_val_ndcg':best['ndcg@10'],'best_first12':first12,'final_epoch':h[-1],'test_metrics':training['test_metrics'],'efficiency_metrics':training['efficiency_metrics']}
(root/'comparison.json').write_text(json.dumps(result,indent=2))
lines=['# Validation evidence summary','','All models use the same 6034 users and 101 candidates per user.','', '| Pre-truncation history | n | Mix50 HR / NDCG | Mix100 HR / NDCG | Mix200 HR / NDCG | Target attention HR / NDCG |','|---|---:|---:|---:|---:|---:|']
for bucket in buckets:
 cells=[]
 for name in scores:
  x=result['metrics'][name][bucket];cells.append(f"{x['hr@10']:.6f} / {x['ndcg@10']:.6f}")
 lines.append(f"| {bucket} | {int(buckets[bucket].sum())} | "+' | '.join(cells)+' |')
lines+=['',f"Baseline best epoch {best['epoch']}, validation NDCG@10 {best['ndcg@10']:.6f}.",'',result['ci_note']]
(root/'comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines));print(json.dumps(result['baseline_training'],indent=2))
