# LLM Evaluation Report — `run4`

Comparison of the LLM's selected answer against the ground-truth answer for all 94 questions. Rationales were ignored. The 11 questions that are not in (a)/(b)/(c) format are graded as five-statement questions, as described in §2.

## 1. Headline results

| Metric | Result |
|---|---|
| **Question-level accuracy (all 94 questions)** | **20 / 94 = 21.3%** |
| Combination-format questions, (a)/(b)/(c) | 83 questions, 16 correct (19.3%) |
| Five-statement questions, ①–⑤ treated as statements | 11 questions, 4 correct (36.4%) |
| **Total statements** | **304** = 249 (3 × 83) + 55 (5 × 11) |
| **Statements judged correctly overall** | **184 / 304 = 60.5%** |
| Statements judged correctly, (a)/(b)/(c) questions | 142 / 249 = 57.0% |
| Statements judged correctly, five-statement questions | 42 / 55 = 76.4% |

## 2. Grading method

**Combination-format questions (83).** Each has statements (a), (b), (c), and each choice ①–⑤ maps to the set of statements judged correct. The mapping differs between questions, so it was read from each question's own choices line. A statement is judged correctly when the LLM's verdict (in the set or not) matches the ground truth's.

**Five-statement questions (11).** These are the questions whose options are full-sentence analyses, numbered statements (1)–(5), or argument-structure graphs. Options ①–⑤ are treated as five statements. In each of these questions the ground truth selects exactly one option. The option the LLM chose is marked true and the other four are marked false. A statement is judged correctly when its verdict matches the ground truth's.

- If the LLM picks the right option, all 5 statements are judged correctly.
- If it picks a wrong option, 2 statements are wrong (the ground-truth option it missed and the option it wrongly chose) and 3 are right, so 3 of 5.
- For “NOT correct” / “cannot be included” questions (2021_03, 2021_25, 2023_22, 2024_05), the selected option is the one the question asks for. Whether that is labelled “true” or “false” does not change the count, because agreement is symmetric.
- 2024_25: the LLM answered with a bare letter (“e”), which is not a valid option and cannot be mapped to one. It is treated as having selected nothing, so all five statements are marked false. Four match the ground truth and the missed ground-truth option does not, giving 4 of 5.

## 3. Statement-level breakdown

### (a)/(b)/(c) questions

| Statement | Judged correctly | Accuracy |
|---|---|---|
| (a) | 52 / 83 | 62.7% |
| (b) | 41 / 83 | 49.4% |
| (c) | 49 / 83 | 59.0% |
| **All** | **142 / 249** | **57.0%** |

### Five-statement questions

| Statement position | Judged correctly | Accuracy |
|---|---|---|
| ① | 9 / 11 | 81.8% |
| ② | 9 / 11 | 81.8% |
| ③ | 7 / 11 | 63.6% |
| ④ | 8 / 11 | 72.7% |
| ⑤ | 9 / 11 | 81.8% |
| **All** | **42 / 55** | **76.4%** |

This 76.4% is not directly comparable with the (a)/(b)/(c) figure, because each of these questions has four false statements, so marking most statements false earns many matches. If the LLM chose one of the five options uniformly at random, the expected score would be 68%.

### Confusion matrix (all 304 statements)

| | GT: correct | GT: not correct |
|---|---|---|
| **LLM: correct** | 107 | 83 |
| **LLM: not correct** | 37 | 77 |

- **(a)/(b)/(c) questions:** TP 103, FP 77, FN 30, TN 39. The LLM marked 180 of 249 statements as correct, while the ground truth marks 133.
- **Five-statement questions:** TP 4, FP 6, FN 7, TN 38.

### Statements correct per question

| Statements correct | (a)/(b)/(c) questions (of 3) | Five-statement questions (of 5) |
|---|---|---|
| 5 | — | 4 |
| 4 | — | 1 |
| 3 | 16 | 6 |
| 2 | 38 | 0 |
| 1 | 18 | 0 |
| 0 | 11 | 0 |

## 4. Breakdown by exam year (question level)

| Year | Correct | Total | Accuracy |
|---|---|---|---|
| 2021 | 4 | 16 | 25.0% |
| 2022 | 2 | 19 | 10.5% |
| 2023 | 8 | 22 | 36.4% |
| 2024 | 2 | 18 | 11.1% |
| 2025 | 4 | 19 | 21.1% |

## 5. Answer-choice distribution (all 94 questions)

| Choice | Ground truth | LLM |
|---|---|---|
| ① | 24 | 0 |
| ② | 24 | 8 |
| ③ | 20 | 50 |
| ④ | 12 | 14 |
| ⑤ | 14 | 21 |
| invalid | 0 | 1 |

The LLM chose ③ in 50 of 94 questions (53.2%), compared with 20 in the ground truth (21.3%).

## 6. Parsing notes and caveats

- **2022_01:** the LLM answered “c” (a statement letter, not an option number). In that question the only option whose set is exactly “(c)” is ②, so it was read as ②. This is wrong either way.
- **2023_09:** the LLM wrote “{④ (a), (b)} Rationale 1. The sta…”, where the circled number and the listed statements disagree (④ means “(a), (c)” in that question, but the text lists “(a), (b)”). The circled number was used. If the listed statements were used instead, this question would score 0 of 3 instead of 2 of 3 (statement total 182 / 304). Question-level correctness is unaffected (wrong either way).
- **2023_17:** the LLM wrote “③ (a), (b), (c).  Rationale 1. Th…”, where the circled number and the listed statements disagree (③ means “(a), (b)” in that question, but the text lists “(a), (b), (c)”). The circled number was used. If the listed statements were used instead, this question would score 2 of 3 instead of 3 of 3 (statement total 183 / 304). Question-level accuracy would lose 1 (20 to 19 of 94), since the listed statements correspond to a different option.
- **2024_24:** the LLM wrote “③ (a), (b), (c)  Rationale 1. If…”, where the circled number and the listed statements disagree (③ means “(a), (b)” in that question, but the text lists “(a), (b), (c)”). The circled number was used. If the listed statements were used instead, this question would score 1 of 3 instead of 2 of 3 (statement total 183 / 304). Question-level correctness is unaffected (wrong either way).
- **2024_25:** if the bare letter were read as option ⑤, this question would score 3 of 5 instead of 4 of 5 (statement total 183 / 304); it stays wrong at question level.
- Several LLM answers appear as `{③}`, `{5}` or `{⑵}`; the number inside the braces was used.
- Question-level “correct” means the LLM's option number equals the ground-truth option number.