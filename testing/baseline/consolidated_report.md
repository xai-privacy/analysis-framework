# Consolidated LLM Evaluation Report (runs 1–5)

Averages across 5 runs of the same LLM on the same 94 benchmark questions, each compared against the same ground truth. All figures are simple means of the five per-run reports. Every run uses the same denominators, so the mean of counts and the mean of percentages are identical. "SD" is the sample standard deviation across the 5 runs.

## 1. Headline results (mean of 5 runs)

| Metric | Mean count | Mean accuracy | SD (pp) | Min – Max |
|---|---|---|---|---|
| **Question-level accuracy (94 questions)** | 19.2 / 94 | **20.4%** | 3.9 | 16.0% – 25.5% |
| (a)/(b)/(c) questions (83) | 16.6 / 83 | 20.0% | 4.3 | 15.7% – 26.5% |
| Five-option questions (11) | 2.6 / 11 | 23.6% | 10.4 | 9.1% – 36.4% |
| **Statement-level accuracy (304 statements)** | 172.8 / 304 | **56.8%** | 4.1 | 52.0% – 60.9% |
| (a)/(b)/(c) statements (249) | 133.6 / 249 | 53.7% | 4.4 | 49.0% – 58.2% |
| Five-option statements (55) | 39.2 / 55 | 71.3% | 4.1 | 65.5% – 76.4% |

Chance baselines for reference:
- Question level: about 20% (5 options).
- (a)/(b)/(c) statements: the ground truth marks 133 of 249 statements (53.4%) as correct, so a guesser who always answers "correct" would score 53.4%.
- Five-option statements: random option choice gives an expected 68%.

## 2. Per-run results

| Run | Question acc. | Statement acc. (304) | (a)/(b)/(c) stmts (249) | Five-option stmts (55) |
|---|---|---|---|---|
| 1 | 15 / 94 (16.0%) | 158 (52.0%) | 122 (49.0%) | 36 (65.5%) |
| 2 | 24 / 94 (25.5%) | 175 (57.6%) | 137 (55.0%) | 38 (69.1%) |
| 3 | 16 / 94 (17.0%) | 162 (53.3%) | 122 (49.0%) | 40 (72.7%) |
| 4 | 20 / 94 (21.3%) | 184 (60.5%) | 142 (57.0%) | 42 (76.4%) |
| 5 | 21 / 94 (22.3%) | 185 (60.9%) | 145 (58.2%) | 40 (72.7%) |
| **Mean** | **19.2 (20.4%)** | **172.8 (56.8%)** | **133.6 (53.7%)** | **39.2 (71.3%)** |

## 3. Statement-level breakdown (mean of 5 runs)

### (a)/(b)/(c) questions

| Statement | Per-run correct (R1–R5) | Mean correct | Mean accuracy |
|---|---|---|---|
| (a) | 44, 45, 41, 52, 55 | 47.4 / 83 | 57.1% |
| (b) | 35, 48, 39, 41, 44 | 41.4 / 83 | 49.9% |
| (c) | 43, 44, 42, 49, 46 | 44.8 / 83 | 54.0% |
| **All** | 122, 137, 122, 142, 145 | **133.6 / 249** | **53.7%** |

### Five-option questions

| Option position | Per-run correct (R1–R5) | Mean correct | Mean accuracy |
|---|---|---|---|
| ① | 9, 9, 9, 9, 9 | 9.0 / 11 | 81.8% |
| ② | 9, 9, 8, 9, 9 | 8.8 / 11 | 80.0% |
| ③ | 3, 5, 6, 7, 6 | 5.4 / 11 | 49.1% |
| ④ | 8, 7, 7, 8, 7 | 7.4 / 11 | 67.3% |
| ⑤ | 7, 8, 10, 9, 9 | 8.6 / 11 | 78.2% |
| **All** | 36, 38, 40, 42, 40 | **39.2 / 55** | **71.3%** |

Option ③ is the weakest position. That is partly because the LLM tends to choose ③, so when it is wrong it is often wrong on ③. The 71.3% is not comparable with the (a)/(b)/(c) figure because four of the five options are always false.

### Confusion matrix (mean per run, 304 statements)

| | GT: correct | GT: not correct |
|---|---|---|
| **LLM: correct** | 97.6 | 84.8 |
| **LLM: not correct** | 46.4 | 75.2 |

| Subset | TP | FP | FN | TN |
|---|---|---|---|---|
| (a)/(b)/(c) questions (249) | 95.0 | 77.4 | 38.0 | 38.6 |
| Five-option questions (55) | 2.6 | 7.4 | 8.4 | 36.6 |

For the (a)/(b)/(c) questions:
- The LLM marks about 172.4 of 249 statements (69.2%) as correct, while the ground truth marks 133 (53.4%). It over-accepts statements.
- Recall on correct statements is 71.4% (95.0 / 133).
- Precision is 55.1% (95.0 / 172.4), barely above the 53.4% base rate.
- Specificity on incorrect statements is 33.3% (38.6 / 116).

## 4. Accuracy by exam year (question level)

| Year | Per-run correct (R1–R5) | Mean correct | Total | Mean accuracy |
|---|---|---|---|---|
| 2021 | 1, 4, 2, 4, 5 | 3.2 | 16 | 20.0% |
| 2022 | 4, 7, 3, 2, 3 | 3.8 | 19 | 20.0% |
| 2023 | 4, 6, 7, 8, 8 | 6.6 | 22 | 30.0% |
| 2024 | 4, 5, 2, 2, 4 | 3.4 | 18 | 18.9% |
| 2025 | 2, 2, 2, 4, 1 | 2.2 | 19 | 11.6% |

2023 is the strongest year and 2025 the weakest, though with only 2 to 7 correct answers per run in a given year, these differences are small relative to run-to-run noise.

## 5. Answer-choice distribution (mean per run, 94 questions)

| Choice | Ground truth | LLM mean | LLM per run (R1–R5) |
|---|---|---|---|
| ① | 24 | 0.4 | 1, 0, 0, 0, 1 |
| ② | 24 | 12.8 | 9, 18, 17, 8, 12 |
| ③ | 20 | 45.0 | 51, 48, 36, 50, 40 |
| ④ | 12 | 14.2 | 11, 12, 17, 14, 17 |
| ⑤ | 14 | 20.4 | 21, 15, 22, 21, 23 |
| invalid | 0 | 1.2 | 1, 1, 2, 1, 1 |

On average the LLM picks ③ in 47.9% of questions (45.0 / 94), against 21.3% in the ground truth. It almost never picks ①, which is the correct answer in 25.5% of questions. This systematic bias in mapping statement verdicts to option numbers is likely a major driver of the low question-level accuracy.

## 6. Parsing notes and caveats

- **2024_25 was invalid in all 5 runs.** The LLM answered with a bare letter ("e", "a", "e", "e", "g"), which is not a valid option. It was treated as no selection and scored 4/5 in every run.
- **2023_11 (run 3 only)** was invalid because the LLM listed three options ("②, ③, ④") in a single-answer question. It was treated as no selection and scored 1/3.
- **Statement letters read as options.** Runs 2 and 3 answered "c" for 2023_18, run 2 answered "c" for 2024_22, and run 4 answered "c" for 2022_01. Each was read as ② because that is the only option whose set is exactly "(c)". Reading them instead as invalid would reduce the affected runs' question and statement totals by 1 each, except 2022_01, which is wrong either way.
- **Circled number vs. listed statements disagree** in a few answers: 2023_09 (runs 1, 4), 2023_17 (run 4), 2024_24 (runs 2, 4). All reports used the circled number. The only case where this changes question-level correctness is 2023_17 in run 4 (−1 question, −1 statement if the listed statements were used).
- **Run 5's report has no per-question parsing notes**, although 2024_25 was invalid in it as well.
- Question-level "correct" means the LLM's option number equals the ground-truth option number.
- "Not correct" / "cannot be included" questions (2021_03, 2021_25, 2023_22, 2024_05) were graded by the option the question asks for. This does not affect counts because agreement is symmetric.
- Only aggregate counts were available in the five reports, not per-question data. Majority-vote accuracy and per-question consistency across runs therefore cannot be computed from these files.

## 7. Key takeaways

1. **Question-level accuracy averages 20.4%, essentially at the 20% chance level.** Runs range from 16.0% to 25.5%.
2. **Statement-level accuracy on (a)/(b)/(c) questions averages 53.7%, about the same as the 53.4% "always correct" baseline.** Runs 4 and 5 (57.0% and 58.2%) are the only ones clearly above it.
3. **The LLM over-accepts statements.** It marks about 69% as correct against a true rate of 53%, which gives it weak discrimination on incorrect statements (specificity about 33%).
4. **There is a strong bias toward option ③** and against ①, in every run.
5. **Run-to-run variability is about ±4 percentage points**, so differences between individual runs or years should be interpreted cautiously.