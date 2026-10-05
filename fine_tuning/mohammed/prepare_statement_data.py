#!/usr/bin/env python3
"""Prepare statement-level datasets using an existing split.

Experiments:
  statement_only     - remove ①..⑤ choices; predict each statement CORRECT/INCORRECT
  statement_renamed  - same, but deterministically replace Alice/Bob/Charlie per question
  compact_structured - labels + deterministic correct-statement aggregation

No test examples are written by this script.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from leet_statement_utils import *

SYSTEM_STATEMENT = """You are solving a LEET-Arg reasoning question.
Use only the information in the passage and statements.
Do not guess from speaker names or answer positions.
For every statement, output exactly one verdict: CORRECT or INCORRECT.
Do not output an answer-choice number."""

SYSTEM_COMPACT = """You are solving a LEET-Arg reasoning question.
Use only the information in the passage and statements.
Do not guess from speaker names or answer positions.
Evaluate every statement as CORRECT or INCORRECT, then list the statement numbers that are correct.
Do not output an answer-choice number."""

def make(q, mode, seed):
    labels, label_mode=derive_gold_statement_labels(q)
    user=strip_choice_numbers(q)
    mapping=None
    if mode=='statement_renamed':
        mapping=deterministic_name_map(str(q['id']),seed)
        user=rename_speakers(user,mapping)
    if mode in ('statement_only','statement_renamed'):
        assistant=labels_to_target(labels); system=SYSTEM_STATEMENT
    elif mode=='compact_structured':
        assistant=labels_to_compact_structured_target(labels); system=SYSTEM_COMPACT
    else: raise ValueError(mode)
    return {
      'id':str(q['id']), 'label_mode':label_mode,
      'gold_statement_labels':labels,
      'name_map':mapping,
      'messages':[
        {'role':'system','content':system},
        {'role':'user','content':user},
        {'role':'assistant','content':assistant},
      ]
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--split',required=True)
    ap.add_argument('--out',default='./data')
    ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args()
    source={str(q['id']):q for q in read_json(args.source)}
    split=read_json(args.split)
    train_ids=[str(x) for x in split['train_ids']]
    valid_ids=[str(x) for x in split['valid_ids']]
    test_ids=set(map(str,split['test_ids']))
    assert not(set(train_ids)&test_ids) and not(set(valid_ids)&test_ids)
    for qid in train_ids+valid_ids:
        if qid not in source: raise SystemExit(f'Missing source ID: {qid}')
        derive_gold_statement_labels(source[qid]) # fail early
    out=Path(args.out)
    for mode in ('statement_only','statement_renamed','compact_structured'):
        write_jsonl(out/mode/'train.jsonl',[make(source[i],mode,args.seed) for i in train_ids])
        write_jsonl(out/mode/'valid.jsonl',[make(source[i],mode,args.seed) for i in valid_ids])
    name_maps={i:deterministic_name_map(i,args.seed) for i in train_ids+valid_ids}
    write_json(out/'statement_renamed'/'name_map.json',name_maps)
    print(json.dumps({'train':len(train_ids),'valid':len(valid_ids),'test_written':0,'output':str(out.resolve())},indent=2))
if __name__=='__main__': main()
