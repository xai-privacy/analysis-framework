#!/usr/bin/env python3
"""Validate statement-level datasets."""
import argparse,json,re
from pathlib import Path
from leet_statement_utils import read_json

def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='./data'); ap.add_argument('--split',default='./data/split.json'); args=ap.parse_args()
    split=read_json(args.split); test=set(map(str,split['test_ids']))
    for mode in ('statement_only','statement_renamed','compact_structured'):
        tr=rows(Path(args.data)/mode/'train.jsonl'); va=rows(Path(args.data)/mode/'valid.jsonl')
        ti={r['id'] for r in tr}; vi={r['id'] for r in va}
        assert not(ti&vi), f'{mode}: train/valid overlap'
        assert not(ti&test), f'{mode}: test leakage in train'
        assert not(vi&test), f'{mode}: test leakage in valid'
        assert len(tr)==len(split['train_ids']) and len(va)==len(split['valid_ids'])
        for r in tr+va:
            user=r['messages'][1]['content']; assistant=r['messages'][2]['content']
            assert '<choices>' not in user, f"{mode}/{r['id']}: choices still present"
            assert not re.search(r'Answer:\s*[1-5①②③④⑤]',assistant,re.I), f"{mode}/{r['id']}: answer number leaked into target"
        print(f'{mode}: PASS train={len(tr)} valid={len(va)}')
    print('No held-out test IDs were written.')
if __name__=='__main__': main()
