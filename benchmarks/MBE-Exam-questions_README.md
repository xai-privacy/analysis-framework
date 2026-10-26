---
language:
- en
pretty_name: "Multistate Bar Exam Legal Questions Dataset"
tags:
- law
- legal
- education
- multistate bar exam
- tabular
license: mit
size_categories:
- 10K<n<100K
---

# Multistate Bar Exam Legal Questions Dataset

This repository contains two datasets of legal questions from the Multistate Bar Examination (MBE) with multiple-choice answers and explanations.

## Dataset Structure

Both datasets share the same structure with the following fields:

| Field | Description |
|-------|-------------|
| `question_number` | Unique identifier for each question |
| `domain` | The area of law the question belongs to |
| `question_text` | The full text of the legal question |
| `options` | A dictionary containing the multiple-choice options |
| `correct_answer` | The correct answer among the available options |
| `explanation` | Justification for the correct answer (format differs between datasets) |

## Datasets Overview

The repository includes two related datasets:

1. **raw_dataset.json**: Contains MBE questions with human-written explanations collected from the internet.
2. **distilled_dataset.json**: Contains the same questions but with explanations restructured according to the Issue, Rule, Application, Conclusion (IRAC) framework using a Llama-3 70B model.

### Legal Domains

Questions span the seven legal domains tested on the Multistate Bar Exam:
- Criminal Law
- Constitutional Law
- Contracts
- Evidence
- Civil Procedure
- Property
- Torts

## Dataset Differences

The key difference between the two datasets lies in the `explanation` field:

- **raw_dataset**: Contains explanations as originally written by humans, collected from various internet sources including study guides and bar exam materials.
- **distilled_dataset**: Contains explanations that have been restructured using the IRAC (Issue, Rule, Application, Conclusion) framework, a standard legal analysis method used by attorneys and law students.

## Source Information

The questions are sourced from past Multistate Bar Examinations (MBE), study guides, and practice materials. No specific metadata about individual questions (such as year or specific source) is included in the datasets.

## Limitations and Biases

This dataset consists of questions from the Multistate Bar Examination, which is designed specifically for the U.S. legal system. The content reflects the focus and priorities of the U.S. legal education system and bar examination, which may not be representative of legal systems in other countries or jurisdictions.

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Dataset"), to deal in the Dataset without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Dataset, and to permit persons to whom the Dataset is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Dataset.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE DATASET.

## Citation

If you use this dataset in your research, please cite:

```
@misc{fernandes2025llamawalksbarefficient,
      title={A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam}, 
      author={Rean Fernandes and André Biedenkapp and Frank Hutter and Noor Awad},
      year={2025},
      eprint={2504.04945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04945}, 
}
```