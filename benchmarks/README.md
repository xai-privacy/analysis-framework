# Benchmarks

## 1. barexam_qa

`barexam_qa_qa.csv` is an unmodified copy of the qa.csv file of the [barexam_qa](https://huggingface.co/datasets/reglab/barexam_qa/tree/main/data/qa) benchmark. The data of test.csv, train.csv, and validation.csv (all available in the barexam_qa benchmark repo, but not included here) seem to be all contained in qa.csv. Per the [barexam_qa readme](https://huggingface.co/datasets/reglab/barexam_qa), the benchmark was created by linking multistate bar exam multiple-choice questions with supporting gold passages from primary sources (caselaw) and secondary sources (online resources, textbooks, etc.). The gold passages were annotated by a law student, simulating a legal research workflow.

Each example in `barexam_qa_qa.csv` consists of a question, multiple-choice answers, an answer label, and a gold passage (passage ID) label. The benchmark contains 1,195 examples.

The labels have the following meaning:

- idx: unique example ID (string)
- prompt: the question prompt (context before the question, can be shared between questions)
- question: the question
- choice_a: answer choice A text
- choice_b: answer choice B text
- choice_c: answer choice C text
- choice_d: answer choice D text
- answer: A, B, C, or D
- gold_passage: gold passage text
- gold_idx: passage ID of gold passage (matches with passage ID in passages)

## 2. LEET-Arg

### 2.1 Files

- `LEET_Arg_Questions_cleaned.json` is a cleaned copy of the LEET-Arg benchmark (Park and Park, [When
  correct is not enough](https://doi.org/10.1016/j.knosys.2026.116625), upstream repo:
  `lit-ai-lab/leet_arg_dataset_v1`, file: `LEET_Arg_Questions.json`. sha256: `f52a9a8f0405192a53990e56eef04ae0fd0d474f049ac6e721255332f8fc3cea`)
- `LEET_Arg_Questions_cleaned_and_rationale_by_statement.json` is based on `LEET_Arg_Questions_cleaned.json` with additional breakdowns of rationales for individual statements
- `LEET_Arg_Model_Responses.json` in this directory holds per-model responses keyed by the same
  `id`/`year`/`problem_idx` and is out of scope here

### 2.2 Contents

93 records, 301 statement units (upstream has 97 / 315 — see "Changes applied" for why). Each record
has: `id`, `year`, `problem_idx`, `objective`, `domain`, `category`, `answer`, `original_question`,
`statements`, `original_rationale`.

- `answer` is a **string** ("1" through "5"), not an integer.
- `statements` has 3 entries for 82 records and 5 entries for 11: 2021_03, 2021_25, 2022_36, 2023_08,
  2023_19, 2023_22, 2023_23, 2024_05, 2024_14, 2024_25, 2025_21.
- `domain` is `null` for 9 records: 2021_36, 2021_37, 2022_38, 2022_39, 2023_35, 2024_37, 2025_26,
  2025_39, 2025_40.
- Answer distribution: 1:24, 2:24, 3:19, 4:12, 5:14 (majority-class baseline 24/93).

### 2.3 Changes applied

#### 2.3.1 `LEET_Arg_Questions_cleaned.json`

Produced by `tools/clean_leet_arg.py`, plus manual edits. `original_rationale`, `answer`, and all
metadata fields are byte-identical to upstream; everything below touches `statements` and/or
`original_question`.

- **Statements re-derived** from `original_question` for 9 records with confirmed upstream
  segmentation errors, using a "next expected label only" rule: 2021_25, 2021_34, 2022_19, 2023_11,
  2023_17, 2023_29, 2024_27, 2025_17, 2025_39. All other records' `statements` are upstream text.
- **2021_13**: corrected `answer` from "4" to "3" per @sungjunleeee's [comment](https://github.com/xai-privacy/analysis-framework/issues/14#issuecomment-5964850636).
- **2025_05**: statements manually reconstructed (upstream had `statements: null`). Hardcoded as
  `MANUAL_2025_05_STATEMENTS` in the script, so future edits go there, not the JSON.
- **2022_31**: `original_question` choice markers normalized from 1.–5. to ①–⑤. Applied by hand, not
  by the script — a clean re-run reverts it.
- **2025_16**: corrected the choices in the `original_question` per @sungjunleeee's [comment](https://github.com/xai-privacy/analysis-framework/issues/14#issuecomment-5964850636).
- **3 records dropped** for inconsistent `original_rationale`, not repaired: 2021_01,
  2021_15, 2023_37.
- **Cosmetic spacing** normalized (`"(a) ."` → `"(a)."`) in 6 records: 2022_08, 2023_29, 2024_06,
  2024_22, 2024_29, 2025_23.

See [issue #14](https://github.com/xai-privacy/analysis-framework/issues/14) for the full cleaning discussion.

#### 2.3.2 `LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`

The changes in `LEET_Arg_Questions_cleaned_and_rationale_by_statement.json` relate to the `original_rationale` field, which we broke down in sub-fields. For example, we reformatted:

```json
    "original_rationale": "Pure risk refers to a risk that has not been realized. The main point of contention in the debate is the moral status of acts that impose pure risk. The views of Alice through Diane are as follows:Alice: All acts that impose pure risk are morally wrong.Bob: An act imposing pure risk can only be considered morally wrong when, had they known of the risk, the agent’s autonomous choice of action could have changed.Charlie: Disagrees with Bob’s criteria, arguing that imposing pure risk on a person in a coma or an infant can also be morally wrong. (Charlie does not present a general principle for when imposing pure risk is morally wrong.)Diane: Only considers it morally wrong if actual incidental harm occurred as a result of the pure risk.<Explanations>(a): Charlie specifically asserts that imposing pure risk on a person in a coma can be morally wrong, and Alice argues that all acts imposing pure risk are morally wrong. Thus, (a) is a correct analysis.(b): Since Alice holds that all acts imposing pure risk are morally wrong, and Bob and Diane only claim that some such acts are morally wrong under certain conditions, Alice would also recognize those acts as morally wrong whenever Bob or Diane do. Therefore, (b) is a correct analysis.(c): Charlie disagreed with Bob’s view only in stating that limiting moral wrongness to the infringement of autonomy is not sufficient, but did not clarify that acts infringing autonomy via pure risk are not wrong. Thus, there is no basis to analyze that Bob and Charlie’s opinions are different regarding this case, so (c) is not a correct analysis.Therefore, only (a) and (b) are correct analyses, so the answer is ③."
```

into:

```json
    "original_rationale": {
      "preliminaries": "Pure risk refers to a risk that has not been realized. The main point of contention in the debate is the moral status of acts that impose pure risk. The views of Alice through Diane are as follows:Alice: All acts that impose pure risk are morally wrong.Bob: An act imposing pure risk can only be considered morally wrong when, had they known of the risk, the agent’s autonomous choice of action could have changed.Charlie: Disagrees with Bob’s criteria, arguing that imposing pure risk on a person in a coma or an infant can also be morally wrong. (Charlie does not present a general principle for when imposing pure risk is morally wrong.)Diane: Only considers it morally wrong if actual incidental harm occurred as a result of the pure risk.",
      "statement_1_rationale": "(a) Charlie specifically asserts that imposing pure risk on a person in a coma can be morally wrong, and Alice argues that all acts imposing pure risk are morally wrong. Thus, (a) is a correct analysis.",
      "statement_2_rationale": "(b) Since Alice holds that all acts imposing pure risk are morally wrong, and Bob and Diane only claim that some such acts are morally wrong under certain conditions, Alice would also recognize those acts as morally wrong whenever Bob or Diane do. Therefore, (b) is a correct analysis.",
      "statement_3_rationale": "(c) Charlie disagreed with Bob’s view only in stating that limiting moral wrongness to the infringement of autonomy is not sufficient, but did not clarify that acts infringing autonomy via pure risk are not wrong. Thus, there is no basis to analyze that Bob and Charlie’s opinions are different regarding this case, so (c) is not a correct analysis.",
      "answer": "Therefore, only (a) and (b) are correct analyses, so the answer is ③."
    }
```

The changes mostly relate to form and not substance. Though, a few changes are noteworthy.

- The following questions did not have any language that we could use to populate the `answer` sub-field. Thus, we added a sentence in the same style as shown in the example above ("Therefore, only (a) and (b) are correct analyses, so the answer is ③.").
  - "id": "2021_03"
  - "id": "2021_25"
  - "id": "2022_12"
  - "id": "2023_03"
  - "id": "2023_08"
  - "id": "2023_20"
  - "id": "2023_22"
  - "id": "2023_23"
  - "id": "2024_05"
  - "id": "2024_14"
  - "id": "2024_22"
  - "id": "2024_27"
- For three questions we kept the `original_rationale` as is because the rationales of the individual statements were all interconnected.
  - "id": "2023_19"
  - "id": "2024_25"
  - "id": "2025_21"
- For question "id": "2023_20" we corrected the answer from "Among the options, only ② (c) is the wrong analysis; the answer should be ②." to "Among the options, only (c) is the wrong analysis; the answer should be ③."
- For question "id": "2023_03" we added "Therefore, (c) is a correct analysis." at the end of `statement_3_rationale`.
- For two questions we changed the formatting as follows.
  - For question "id": "2022_31" from "Because only (c) is a correct evaluation, the answer is 2." to "Because only (c) is a correct evaluation, the answer is ②."
  - For question "id": "2025_02"from "Only (b) is a correct analysis from the given choices, so the correct answer is 1." to "Only (b) is a correct analysis from the given choices, so the correct answer is ①."
- We removed markers in the original_rationale to identify beginnings of sections that were no longer necessary, such as "Explanation.", to identify the beginning of the discussion of the individual statement rationales.
- We fixed some formatting that was not uniform in the original `original_rationale`, e.g., we changed "(a)." to "(a)".

See [issue #17](https://github.com/xai-privacy/analysis-framework/issues/17) for further details.

## 3.MBE-exam-questions

### 3.1 Files

`MBE-exam-questions_distilled_dataset.json`, `MBE-exam-questions_raw_dataset.json`, and `MBE-Exam-questions_README` are unmodified copies of the respective files in the [MBE-exam-questions](https://huggingface.co/datasets/HolySaint/MBE-exam-questions/tree/main) benchmark repo. See the `MBE-Exam-questions_README` for details on this benchmark.
