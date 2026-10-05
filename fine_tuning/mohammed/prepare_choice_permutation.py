#!/usr/bin/env python3
"""Create a diagnostic direct-answer dataset with per-question choice order permuted.

The logical question is unchanged. Only the order/number assigned to the five
choices changes. This tests whether the model learns arbitrary answer positions.
Uses only existing train/validation split; writes no test data.
"""
from __future__ import annotations
import argparse,json,random,hashlib
from pathlib import Path
from leet_statement_utils import *
SYSTEM="""You are solving a LEET-Arg reasoning question. Use only the information in the question. Choose the correct option. End with exactly: Answer: <number from 1 to 5>."""

def perm_for(qid,seed):
    h=hashlib.sha256(f'choice:{seed}:{qid}'.encode()).hexdigest(); rng=random.Random(int(h[:16],16)); p=list(range(5)); rng.shuffle(p); return p

def rewrite(q,seed):
    base,ch=split_choices(q['original_question']); opts=parse_choice_options(ch)
    if len(opts)!=5: raise ValueError(f"{q['id']}: expected 5 choices, got {len(opts)}")
    p=perm_for(str(q['id']),seed) # new position j takes old option p[j]
    new_choices=' '.join(f'{CIRCLED[j]} {opts[p[j]][1]}' for j in range(5))
    old_gold=int(q['answer'])-1; new_gold=p.index(old_gold)+1
    return base+' <choices> '+new_choices,new_gold,p

def rec(q,seed):
    user,gold,p=rewrite(q,seed)
    return {'id':str(q['id']),'original_gold':int(q['answer']),'permuted_gold':gold,'permutation_zero_based':p,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user},{'role':'assistant','content':f'Answer: {gold}'}]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--split',required=True); ap.add_argument('--out',default='./data/choice_permuted'); ap.add_argument('--seed',type=int,default=7); args=ap.parse_args()
    src={str(q['id']):q for q in read_json(args.source)}; sp=read_json(args.split); test=set(map(str,sp['test_ids'])); tr=list(map(str,sp['train_ids'])); va=list(map(str,sp['valid_ids'])); assert not(set(tr+va)&test)
    out=Path(args.out); write_jsonl(out/'train.jsonl',[rec(src[i],args.seed) for i in tr]); write_jsonl(out/'valid.jsonl',[rec(src[i],args.seed) for i in va]); write_json(out/'permutation_map.json',{i:{'permutation_zero_based':perm_for(i,args.seed),'original_gold':int(src[i]['answer']),'permuted_gold':rewrite(src[i],args.seed)[1]} for i in tr+va})
    print(json.dumps({'train':len(tr),'valid':len(va),'test_written':0},indent=2))
if __name__=='__main__': main()
