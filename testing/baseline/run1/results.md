# LLM Evaluation Report — `run1` (revised)

Comparison of the LLM's selected answer against the ground-truth answer for all 94 questions. Rationales were ignored for now.

## 1. Headline results

| Metric | Result |
|---|---|
| **Question-level accuracy (all 94 questions)** | **15 / 94 = 16.0%** |
| Combination-format questions, (a)/(b)/(c) | 83 questions, 14 correct (16.9%) |
| Five-statement questions, ①–⑤ treated as statements | 11 questions, 1 correct (9.1%) |
| **Total statements** | **304** = 249 (3 × 83) + 55 (5 × 11) |
| **Statements judged correctly overall** | **158 / 304 = 52.0%** |
| Statements judged correctly, (a)/(b)/(c) questions | 122 / 249 = 49.0% |
| Statements judged correctly, five-statement questions | 36 / 55 = 65.5% |

Question-level accuracy is unchanged by the new rule. Only the statement-level counts change.

## 2. Grading method

**Combination-format questions (83).** Each has statements (a), (b), (c), and each choice ①–⑤ maps to the set of statements judged correct. The mapping differs between questions, so it was read from each question's own choices line. A statement is judged correctly when the LLM's verdict (in the set or not) matches the ground truth's.

**Five-statement questions (11).** These are the questions whose options are full-sentence analyses, numbered statements (1)–(5), or argument-structure graphs. In each of these questions the ground truth selects exactly one option. The option the LLM chose is marked true and the other four are marked false. A statement is judged correctly when its verdict matches the ground truth's.

- If the LLM picks the right option, all 5 statements are judged correctly.
- If it picks a wrong option, 2 statements are wrong (the ground-truth option it missed and the option it wrongly chose) and 3 are right, so 3 of 5.
- For “NOT correct” / “cannot be included” questions (2021_03, 2021_25, 2023_22, 2024_05), the selected option is the one the question asks for. Whether that is labelled “true” or “false” does not change the count, because agreement is symmetric.
- 2024_25: the LLM answered “e”, which is not a valid option. It is treated as having selected nothing, so all five statements are marked false. Four match the ground truth and the missed ground-truth option does not, giving 4 of 5.

## 3. Statement-level breakdown

### (a)/(b)/(c) questions

| Statement | Judged correctly | Accuracy |
|---|---|---|
| (a) | 44 / 83 | 53.0% |
| (b) | 35 / 83 | 42.2% |
| (c) | 43 / 83 | 51.8% |
| **All** | **122 / 249** | **49.0%** |

### Five-statement questions

| Statement position | Judged correctly | Accuracy |
|---|---|---|
| ① | 9 / 11 | 81.8% |
| ② | 9 / 11 | 81.8% |
| ③ | 3 / 11 | 27.3% |
| ④ | 8 / 11 | 72.7% |
| ⑤ | 7 / 11 | 63.6% |
| **All** | **36 / 55** | **65.5%** |

This 65.5% looks higher than the (a)/(b)/(c) figure because each of these questions has four false statements, so marking most statements false earns many matches. If the LLM chose one of the five options uniformly at random, the expected score would be 68%, so the LLM is slightly below a random guess here too.

### Confusion matrix (all 304 statements)

| | GT: correct | GT: not correct |
|---|---|---|
| **LLM: correct** | 92 | 94 |
| **LLM: not correct** | 52 | 66 |

- **(a)/(b)/(c) questions:** TP 91, FP 85, FN 42, TN 31. The LLM marked 176 of 249 statements as correct, while the ground truth marks 133. It over-accepts statements.
- **Five-statement questions:** TP 1, FP 9, FN 10, TN 35.

### Statements correct per question

| Statements correct | (a)/(b)/(c) questions (of 3) | Five-statement questions (of 5) |
|---|---|---|
| 5 | — | 1 |
| 4 | — | 1 |
| 3 | 14 | 9 |
| 2 | 31 | 0 |
| 1 | 18 | 0 |
| 0 | 20 | 0 |

## 4. Breakdown by exam year (question level)

| Year | Correct | Total | Accuracy |
|---|---|---|---|
| 2021 | 1 | 16 | 6.2% |
| 2022 | 4 | 19 | 21.1% |
| 2023 | 4 | 22 | 18.2% |
| 2024 | 4 | 18 | 22.2% |
| 2025 | 2 | 19 | 10.5% |

## 5. Answer-choice distribution (all 94 questions)

| Choice | Ground truth | LLM |
|---|---|---|
| ① | 24 | 1 |
| ② | 24 | 9 |
| ③ | 20 | 51 |
| ④ | 12 | 11 |
| ⑤ | 14 | 21 |
| invalid | 0 | 1 |

The LLM chose ③ in 51 of 94 questions (54.3%), compared with 20 in the ground truth (21.3%).

## 6. Parsing notes and caveats

- **2023_09:** the LLM wrote “Answer-⑤ (a), (b)”, but in that question ⑤ means “(a), (b), (c)”. The circled number was used (⑤). Under the alternative reading, the statement total would be 157 / 304 = 51.6% instead of 158 / 304. The question is wrong either way.
- **2024_25:** if “e” were read as option ⑤ instead of as no selection, that question would score 3 of 5 instead of 4 of 5, and the overall statement total would be 157 / 304 = 51.6%.
- Several LLM answers appear as `{③}` or `{5}`. The number inside the braces was used. Answers without an “Answer-” prefix were handled the same way.
- Question-level “correct” means the LLM's option number equals the ground-truth option number.