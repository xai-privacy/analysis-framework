# Mohammed — Preliminary Abstract Argumentation Framework Testing

This folder contains Markdown templates for Sebastian's ten fixed preliminary-testing questions:

- 2021_02
- 2021_14
- 2021_16
- 2021_17
- 2021_24
- 2025_25
- 2025_29
- 2025_30
- 2025_39
- 2025_40

Each question file includes:

- LEET-Arg question ID and metadata
- Full LEET-Arg question text
- Gold answer and original rationale
- Baseline prompt
- Three-step argumentation framework prompts
- Placeholders for manually copied model responses
- Analysis sections for comparing baseline vs argumentation-framework behavior

## Manual testing protocol

For each question, manually run:

1. The baseline prompt.
2. Argumentation Framework Prompt 1: extract arguments.
3. Argumentation Framework Prompt 2: model attack relations.
4. Argumentation Framework Prompt 3: evaluate the answer statements using the extracted arguments and attacks.

Then paste all responses into the corresponding Markdown file and complete the analysis sections.

## Recommended first pass

Use `gpt-4.1-mini` for all ten questions first. Then optionally run `claude-sonnet-4-6` on the most interesting 2–3 cases for comparison.
