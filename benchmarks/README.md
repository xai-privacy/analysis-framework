# Benchmarks

## 1. LEET-Arg

### 1.1 Files

- `LEET_Arg_Questions_cleaned.json` is a cleaned copy of the LEET-Arg benchmark (Park and Park, [When
  correct is not enough](https://doi.org/10.1016/j.knosys.2026.116625), upstream repo:
  `lit-ai-lab/leet_arg_dataset_v1`, file: `LEET_Arg_Questions.json`. sha256: `f52a9a8f0405192a53990e56eef04ae0fd0d474f049ac6e721255332f8fc3cea`).
- `LEET_Arg_Questions_cleaned_and_rationale_by_statement.json` is based on `LEET_Arg_Questions_cleaned.json` with additional breakdowns of rationales for individual statements.
- `LEET_Arg_Questions_cleaned_and_rationale_by_statement_baseline.json` is based on `LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`. Each JSON object contains only `id` and `original_question`. In particular, `original_rationale`s are excluded. For most questions the last sentence with the answer \<choices\> is removed. Only questions 2021_03, 2021_25, 2022_36, 2023_19, 2024_05, 2024_14, 2024_25, and 2025_21 retains the answer \<choices\> sentence.
- `LEET_Arg_Model_Responses.json` in this directory holds per-model responses keyed by the same
  `id`/`year`/`problem_idx` and is out of scope here.

### 1.2 Contents

93 records, 301 statement units (upstream has 97 / 315 — see "Changes applied" for why). Each record
has: `id`, `year`, `problem_idx`, `objective`, `domain`, `category`, `answer`, `original_question`,
`statements`, `original_rationale`.

- `answer` is a **string** ("1" through "5"), not an integer.
- `statements` has 3 entries for 82 records and 5 entries for 11: 2021_03, 2021_25, 2022_36, 2023_08,
  2023_19, 2023_22, 2023_23, 2024_05, 2024_14, 2024_25, 2025_21.
- `domain` is `null` for 9 records: 2021_36, 2021_37, 2022_38, 2022_39, 2023_35, 2024_37, 2025_26,
  2025_39, 2025_40.
- Answer distribution: 1:24, 2:24, 3:19, 4:12, 5:14 (majority-class baseline 24/93).

### 1.3 Changes applied

#### 1.3.1 `LEET_Arg_Questions_cleaned.json`

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

#### 1.3.2 `LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`

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

## 2.MBE-exam-questions

### 2.1 Files

- `MBE-exam-questions_distilled_dataset.json`, `MBE-exam-questions_raw_dataset.json`, and `MBE-Exam-questions_README` are unmodified copies of the respective files in the [MBE-exam-questions](https://huggingface.co/datasets/HolySaint/MBE-exam-questions/tree/main) benchmark repo. See the `MBE-Exam-questions_README` for details on this benchmark.
- `MBE-exam-questions_distilled_dataset_cleaned.json` is based on `MBE-exam-questions_distilled_dataset.json` with duplicates removed and other fixes. [Issue 32](https://github.com/xai-privacy/analysis-framework/issues/32) describes details.
- `MBE-exam-questions_distilled_dataset_cleaned_baseline.json` contains 1,280 records derived from `MBE-exam-questions_distilled_dataset_cleaned.json`. Each JSON object contains `id`, `question`, and the four answer options (`optionA`–`optionD`); correct answers and rationales are excluded. It is an answer-free inference input, not a duplicate of the full source data.

Both baseline files use the same question-only format for inference. Gold answers and rationales stay in the source benchmark files and are used separately for evaluation; they are not part of the baseline inputs.

## 3. LegalBench

### 3.1 Files

- `LegalBench_legal_reasoning_causality_test.tsv` and `LegalBench_legal_reasoning_causality_train.tsv` are the original `train.tsv` and `test.tsv` files from LegalBench’s [legal reasoning causality task](https://huggingface.co/datasets/nguha/legalbench/tree/main/data/legal_reasoning_causality).
- `LegalBench_privacy_policy_entailment_test.tsv` and `LegalBench_privacy_policy_entailment_train.tsv` are the original `train.tsv` and `test.tsv` files from LegalBench’s [privacy policy entailment task](https://huggingface.co/datasets/nguha/legalbench/tree/main/data/privacy_policy_entailment).
- `LegalBench_legal_reasoning_causality_cleaned.json` and `LegalBench_privacy_policy_entailment_cleaned.json` merge the train and test files of each task into one file, renumber `index` from 0, and clean the text. They are made by `fine_tuning/jun/legalbench/clean_legalbench.py`. See [issue #33](https://github.com/xai-privacy/analysis-framework/issues/33).
- `LegalBench_legal_reasoning_causality_cleaned_baseline.json` and `LegalBench_privacy_policy_entailment_cleaned_baseline.json` are answer-free baseline inputs derived from the corresponding cleaned files.

### 3.2 Contents

Each cleaned record has the TSV columns in TSV order, with `index` renumbered across the merged file (train rows first), plus `original_split` and `original_index`, which point back to the source row:

- causality: `index`, `answer`, `text`, `original_split`, `original_index`
- privacy policy entailment: `index`, `text`, `description`, `answer`, `original_split`, `original_index`

The baseline files retain the task inputs and `index`, plus `original_split` for train/test filtering; they omit `answer` and `original_index`:

- causality baseline: `index`, `text`, `original_split`
- privacy policy entailment baseline: `index`, `text`, `description`, `original_split`

| Benchmark                 | Train | Test  | Merged | After cleaning | Labels                       |
| ------------------------- | ----- | ----- | ------ | -------------- | ---------------------------- |
| legal_reasoning_causality | 4     | 55    | 59     | 58             | Yes 32, No 26                |
| privacy_policy_entailment | 8     | 4,335 | 4,343  | 4,341          | Correct 3,743, Incorrect 598 |

### 3.3 Changes applied

Answers were never changed. Counts are records changed, after duplicates were dropped. `fine_tuning/jun/legalbench/legalbench_cleaning_log.json` lists the indices for every change, and `fine_tuning/jun/legalbench/legalbench_review_desk.html` shows each record before and after cleaning.

Causality (`text`):

- Removed court-reporter page markers such as `[*27]` or `[**6]` (54 records). They mark where a page starts in the printed reporter volume.
- Removed legal-database headnote markers such as `HN26` (13 records). They point to editor-written summaries that are not part of the excerpts.
- Whitespace: removed spaces and line breaks at the start or end (20), collapsed repeated spaces (1), and removed spaces around line breaks (1). Paragraph breaks inside the text are kept (16 records).

Privacy policy entailment:

- `description` typo: “by by ad networks” became “by ad networks” (907 records, from 27 of the 57 description sentences).
- `text`, HTML: in 3 records, `<a href="...">http://www.google.com/privacy_<wbr />ads.html</a>` became `http://www.google.com/privacy_ads.html`.
- `text`, PDF ligature characters `ﬁ` and `ﬂ` became `fi` and `fl` (6 records).
- `text`, whitespace: non-breaking spaces became normal spaces (11), and invisible characters such as U+200E were removed (18).

Duplicates dropped (each had the same answer as the record that was kept):

- causality test 29 = test 4 (only a trailing line break differed).
- privacy test 1004 = test 1001 (only “you” vs “You” at the start differed).
- privacy test 2565 = test 579 (only a `ﬁ` ligature in “identiﬁcation” differed, so the two became identical after the ligature fix).

Every output record was compared word by word with its source TSV row; the only differences are the ones listed above. The review desk runs the same cleaning in the browser and gives byte-identical output.

### 3.4 Notes for experiments

- **Train rows are LegalBench’s few-shot examples.** Each task’s `base_prompt.txt` in the [LegalBench repo](https://github.com/HazyResearch/legalbench) is an instruction, a fixed set of solved examples and a slot for the excerpt being evaluated. The solved examples are the train rows (4 for causality, 8 for privacy policy entailment, which are 4 clauses with 2 descriptions each), in the same order and with the same answers. If a one-shot prompt uses one of these rows as its example, that row should not also be scored. Filter on `original_split`.
- **No instruction field.** The TSVs have no question text, because LegalBench keeps the instruction in `base_prompt.txt`. Our prompt has to supply it (issue #34).
- **Upstream size.** LegalBench’s task README says 4,385 samples for privacy policy entailment, but the files here have 4,343 rows. These files have not been compared with the current Hugging Face download yet.
- **Left as in the source:** 7 privacy records (copies of the same Google AdSense clause) write the URL as `http://www.google.com/ privacy_ads.html`, with a space. That is how the policy text was extracted, so it was kept.
