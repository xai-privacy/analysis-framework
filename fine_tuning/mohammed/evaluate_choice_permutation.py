#!/usr/bin/env python3
"""Evaluate the choice-permutation diagnostic on validation IDs."""
from __future__ import annotations
import argparse,json,re,time
from pathlib import Path
from collections import Counter
from mlx_lm import load,generate
from leet_statement_utils import *
from prepare_choice_permutation import rewrite,SYSTEM
CIRC_MAP={'①':1,'②':2,'③':3,'④':4,'⑤':5}
def parse(text):
    m=list(re.finditer(r'(?i)Answer\s*[:：-]\s*([1-5①②③④⑤])',text))
    if m:
        t=m[-1].group(1); return CIRC_MAP.get(t,int(t) if t.isdigit() else None)
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if lines and lines[0][:1] in CIRC_MAP:return CIRC_MAP[lines[0][0]]
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--ids',required=True); ap.add_argument('--adapter-path'); ap.add_argument('--model',default='mlx-community/Llama-3.2-3B-Instruct-4bit'); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--out',required=True); ap.add_argument('--max-tokens',type=int,default=500); args=ap.parse_args()
    src={str(q['id']):q for q in read_json(args.source)}; ids=list(map(str,read_json(args.ids))); model,tok=load(args.model,adapter_path=args.adapter_path); recs=[]
    for j,i in enumerate(ids,1):
        user,gold,p=rewrite(src[i],args.seed); prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':user}],add_generation_prompt=True,tokenize=False); t=time.time(); out=generate(model,tok,prompt=prompt,max_tokens=args.max_tokens,verbose=False); pred=parse(out); recs.append({'id':i,'permuted_gold':gold,'pred_answer':pred,'correct':pred==gold,'permutation_zero_based':p,'seconds':round(time.time()-t,2),'output':out}); print(f'[{j}/{len(ids)}] {i}: pred={pred} gold={gold} correct={pred==gold}')
    summary={'n':len(recs),'accuracy':sum(r['correct'] for r in recs)/len(recs),'parse_rate':sum(r['pred_answer'] is not None for r in recs)/len(recs),'prediction_counts':dict(Counter(r['pred_answer'] for r in recs if r['pred_answer'] is not None))}; Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'summary':summary,'records':recs},ensure_ascii=False,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
