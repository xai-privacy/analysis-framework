# LLM Evaluation Report — `run3`

Comparison of the LLM's selected answer against the ground-truth answer for all 94 questions. Rationales were ignored. The 11 questions that are not in (a)/(b)/(c) format are graded as five-statement questions, as described in §2.

## 1. Headline results

| Metric | Result |
|---|---|
| **Question-level accuracy (all 94 questions)** | **16 / 94 = 17.0%** |
| Combination-format questions, (a)/(b)/(c) | 83 questions, 13 correct (15.7%) |
| Five-statement questions, ①–⑤ treated as statements | 11 questions, 3 correct (27.3%) |
| **Total statements** | **304** = 249 (3 × 83) + 55 (5 × 11) |
| **Statements judged correctly overall** | **162 / 304 = 53.3%** |
| Statements judged correctly, (a)/(b)/(c) questions | 122 / 249 = 49.0% |
| Statements judged correctly, five-statement questions | 40 / 55 = 72.7% |

## 2. Grading method

**Combination-format questions (83).** Each has statements (a), (b), (c), and each choice ①–⑤ maps to the set of statements judged correct. The mapping differs between questions, so it was read from each question's own choices line. A statement is judged correctly when the LLM's verdict (in the set or not) matches the ground truth's.

**Five-statement questions (11).** These are the questions whose options are full-sentence analyses, numbered statements (1)–(5), or argument-structure graphs. Options ①–⑤ are treated as five statements. In each of these questions the ground truth selects exactly one option. The option the LLM chose is marked true and the other four are marked false. A statement is judged correctly when its verdict matches the ground truth's.

- If the LLM picks the right option, all 5 statements are judged correctly.
- If it picks a wrong option, 2 statements are wrong (the ground-truth option it missed and the option it wrongly chose) and 3 are right, so 3 of 5.
- For “NOT correct” / “cannot be included” questions (2021_03, 2021_25, 2023_22, 2024_05), the selected option is the one the question asks for. Whether that is labelled “true” or “false” does not change the count, because agreement is symmetric.
- 2024_25: the LLM answered with a bare letter (“e”), which is not a valid option and cannot be mapped to one. It is treated as having selected nothing, so all five statements are marked false. Four match the ground truth and the missed ground-truth option does not, giving 4 of 5.
- 2023_11: the LLM answered “②, ③, ④”, three options in a single-answer question, which is not a valid selection. It is treated as having selected nothing, so statements (a), (b), (c) are all marked false. That scores 1 of 3 against the ground truth.

## 3. Statement-level breakdown

### (a)/(b)/(c) questions

| Statement | Judged correctly | Accuracy |
|---|---|---|
| (a) | 41 / 83 | 49.4% |
| (b) | 39 / 83 | 47.0% |
| (c) | 42 / 83 | 50.6% |
| **All** | **122 / 249** | **49.0%** |

### Five-statement questions

| Statement position | Judged correctly | Accuracy |
|---|---|---|
| ① | 9 / 11 | 81.8% |
| ② | 8 / 11 | 72.7% |
| ③ | 6 / 11 | 54.5% |
| ④ | 7 / 11 | 63.6% |
| ⑤ | 10 / 11 | 90.9% |
| **All** | **40 / 55** | **72.7%** |

This 72.7% is not directly comparable with the (a)/(b)/(c) figure, because each of these questions has four false statements, so marking most statements false earns many matches. If the LLM chose one of the five options uniformly at random, the expected score would be 68%.

### Confusion matrix (all 304 statements)

| | GT: correct | GT: not correct |
|---|---|---|
| **LLM: correct** | 90 | 88 |
| **LLM: not correct** | 54 | 72 |

- **(a)/(b)/(c) questions:** TP 87, FP 81, FN 46, TN 35. The LLM marked 168 of 249 statements as correct, while the ground truth marks 133.
- **Five-statement questions:** TP 3, FP 7, FN 8, TN 37.

### Statements correct per question

| Statements correct | (a)/(b)/(c) questions (of 3) | Five-statement questions (of 5) |
|---|---|---|
| 5 | — | 3 |
| 4 | — | 1 |
| 3 | 13 | 7 |
| 2 | 28 | 0 |
| 1 | 27 | 0 |
| 0 | 15 | 0 |

## 4. Breakdown by exam year (question level)

| Year | Correct | Total | Accuracy |
|---|---|---|---|
| 2021 | 2 | 16 | 12.5% |
| 2022 | 3 | 19 | 15.8% |
| 2023 | 7 | 22 | 31.8% |
| 2024 | 2 | 18 | 11.1% |
| 2025 | 2 | 19 | 10.5% |

## 5. Answer-choice distribution (all 94 questions)

| Choice | Ground truth | LLM |
|---|---|---|
| ① | 24 | 0 |
| ② | 24 | 17 |
| ③ | 20 | 36 |
| ④ | 12 | 17 |
| ⑤ | 14 | 22 |
| invalid | 0 | 2 |

The LLM chose ③ in 36 of 94 questions (38.3%), compared with 20 in the ground truth (21.3%).

## 6. Parsing notes and caveats

- **2023_18:** the LLM answered “c” (a statement letter, not an option number). In that question the only option whose set is exactly “(c)” is ②, so it was read as ②. This is correct (ground truth ②). If it were instead treated as an invalid answer, question accuracy would lose 1 and the statement total would lose 1.
- **2023_11:** if the three listed options were read as a union of their statements (②, ③, ④ → (a), (b), (c)), this question would score 2 of 3 instead of 1 of 3 (statement total 163 / 304). It stays wrong at question level either way.
- **2024_25:** if the bare letter were read as option ⑤, this question would score 3 of 5 instead of 4 of 5 (statement total 161 / 304); it stays wrong at question level.
- Several LLM answers appear as `{③}`, `{5}` or `{⑵}`; the number inside the braces was used.
- Question-level “correct” means the LLM's option number equals the ground-truth option number.