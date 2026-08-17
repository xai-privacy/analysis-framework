# Analysis Framework

The scripts and other artifacts in this repo are for probing a generative AI model for causal reasoning in legal contexts. For the time being, this is just a small proof-of concept for one scenario, which is a patentee claiming lost profit damages. In this scenario there are two causal considerations a model must make:

1. There needs to be an infringing product; without it there can be no damages
2. There must not be a non-infringing substitute product; if such exists, consumers would use that product instead of the patentee's product

## New workflow: structured predicates + rule application

The repository now uses a two-stage workflow instead of asking the model to reason directly over the full text using the DSL as a system prompt.

1. The model is prompted to emit structured predicate JSON describing the scenario, for example:
   - `infringing_product_available: true|false`
   - `substitute_product_available: true|false`
2. The repo parses that output with Pydantic into a validated schema in `structured_outputs.py`.
3. A small DSL-style rule layer in the same module applies the legal logic and returns a final `AWARDED` or `DENIED` decision.

In other words, the model now acts primarily as a structured extractor, while the repository handles deterministic rule application. This makes the pipeline easier to audit and more compatible with later work on richer formal DSLs.

Here are the setup instructions for running an analysis of this scenario for the Llama 3.2 1B (1 Billion parameters) model. All instructions are for macOS (and the Fish shell).

## 0. Preliminaries

1. Create a [Hugging Face](https://huggingface.co) account.
2. Log into Hugging Face and request access to the [Llama-3.2 1B model](https://huggingface.co/meta-llama/Llama-3.2-1B). It should take only 30 minutes or so.
3. Create an access token on Hugging Face.
4. Install the [Hugging Face command line tools](https://huggingface.co/docs/huggingface_hub/en/guides/cli) with:

   ```bash
   brew install hf
   ```

   Check with:

   ```bash
   hf version
   ```

   if it worked.

5. Log into Hugging Face with:

   ```bash
   hf auth login
   ```

   If asked, you can add the token as git credential.

6. Install Python and create a Python virtual environment (which is not necessary, but makes dependency management easier) with (assuming you have Homebrew installed):

   ```bash
   /opt/homebrew/bin/python3 -m venv .venv
   ```

   Start the virtual environment with:

   ```bash
   source .venv/bin/activate.fish
   ```

   or just `source .venv/bin/activate` if you are not using Fish. You can stop the virtual environment with `deactivate`.

7. Install all dependencies with:

   ```bash
   pip3 install torch transformers accelerate pyvene transformer-lens pydantic
   ```

## 1. Audit

1. Run the baseline audit of the model with:

   ```bash
   python3 run_benchmark.py
   ```

   You can select a different model with `--model` and a different rule DSL with `--dsl`:

   ```bash
   # Use a specific model (default: meta-llama/Llama-3.2-1B-Instruct)
   python3 run_benchmark.py --model microsoft/Phi-4-mini-instruct

   # Use LegalRuleML rules instead of the default plain English
   python3 run_benchmark.py --dsl legalruleml

   # Use De Jure structured rules
   python3 run_benchmark.py --dsl de_jure

   # Use ODRL policy rules
   python3 run_benchmark.py --dsl odrl

   # Combine model and DSL
   python3 run_benchmark.py --model microsoft/Phi-4-mini-instruct --dsl legalruleml
   ```

   **`--model`** accepts any Hugging Face model id for a dense text decoder model (not MoE/multimodal). Examples:
   - `meta-llama/Llama-3.2-1B-Instruct` (default)
   - `Qwen/Qwen3-4B`
   - `microsoft/Phi-4-mini-instruct`

   **`--dsl`** selects the formal rule language embedded in the system prompt:
   - `plain` (default) -- plain English rules (no external file)
   - `odrl` -- loads rules from `odrl_rules.json`
   - `legalruleml` -- loads rules from `legal_rules.xml`
   - `de_jure` -- loads rules from `de_jure_rules.json`

   **Model configs**: Each model's inference settings (dtype, trust_remote_code, seed, generation params) are defined in per-model JSON files under `model_configs/`. The file name is derived by replacing `/` with `_` in the model id (e.g., `microsoft/Phi-4-mini-instruct` -> `model_configs/microsoft_Phi-4-mini-instruct.json`). It is recommended to create a config file for any new model you pass via `--model`. If no config file exists for the given model, the system defaults to the Llama config (`model_configs/meta-llama_Llama-3.2-1B-Instruct.json`). Note that some models require specific settings (e.g., `trust_remote_code: true` or `bfloat16` dtype) and will fail at load time without a matching config file.

2. From the output we see that the model has problems reasoning causally in the patent damages scenario. We can look inside the hidden layers of the model to extract the mathematical concept vector, `ip_concept_vector.pt`, with:

   ```bash
   python3 probe_activations.py
   ```

## 2. Mitigation

Before re-training the model, fine-tuning it, or attempting other mitigations, we can try to modify the responsible model vector on the fly with:

```bash
python3 steer_inference.py
```

However, from the output we see that does not work. The model is still not reasoning correctly.

I can essentially see the following options for fixing a model:

1. Re-training
2. Guardrails
3. Integrating a solver into the model architecture

## 3. Post-intervention Verification

We can rerun the audit under 1. to verify that any permanent verification was successful.

## 4. Limitations

One limitation I ran into is that the model refused to respond to prompts in employment scenarios, e.g., age discrimination. There are built-in safeguards that, when triggered, make the model refuse to respond to prompts. So, either we would need to find enough scenarios that a model can answer or remove the safeguards (but this latter approach is a whole research project on its own, and we may also inadvertently change the model behavior invalidating our findings as those are no longer for the unchanged model).

---

# LEET-Arg argument reasoning harness

The `leet_arg/` package is a separate workstream from the patent damages proof of concept above. It evaluates models on **LEET-Arg**, a legal argument reasoning benchmark, and is the scaffolding for a 2x2 evaluation: local SLM vs frontier API, crossed with plain model vs model+solver, each condition run five times.

**What is built today is one cell of that 2x2: the plain-model, local-SLM baseline.** The other three cells plug into the same stages.

## Pipeline

Six stages with fixed signatures, so the conditions can be swapped independently:

```
Record -> build_prompt() -> ModelAdapter.generate() -> parse() -> Reasoner.reason() -> score()
```

| File | Stage |
| --- | --- |
| `leet_arg/data.py` | `Record` dataclass, loader, fail-loud validation |
| `leet_arg/gold.py` | `<choices>` parsing, stem polarity, gold label derivation |
| `leet_arg/prompts.py` | `build_prompt(record, decomposition=...)` |
| `leet_arg/adapters.py` | `ModelAdapter` protocol, `HFAdapter`, `APIAdapter` (stub), `StubAdapter` |
| `leet_arg/parse.py` | raw model text -> `ParsedAnswer` |
| `leet_arg/reason.py` | `Reasoner` protocol, `PassthroughReasoner` |
| `leet_arg/score.py` | metrics and aggregation |
| `run_leet_arg.py` | CLI entry point |

`PassthroughReasoner` returns the choice the model already picked. It is a real stage in the call path rather than dead code: the solver condition is added later by supplying a different `Reasoner`, and the frontier condition by supplying a different `ModelAdapter`. No other stage changes.

## Running it

```bash
python3 run_leet_arg.py --model meta-llama/Llama-3.2-1B-Instruct --n 10 --trials 5
```

Useful flags:

- `--n` -- records to sample; `-1` runs all 97. The sample is stratified so that both structural shapes are always exercised.
- `--trials` -- repeats per record, each with a distinct seed (`--seed-base` + trial index).
- `--temperature` -- must be greater than zero, and this is enforced. The committed model configs set `do_sample: false`, and greedy decoding would make all five trials identical and the consistency metric meaningless. Default `0.7`.
- `--max-new-tokens` -- default `256`. The prompt asks for a single `ANSWER: <n>` line; this cap keeps a demo run fast. It affects results, so it is recorded in the run summary.
- `--decomposition` -- `choice_level` (implemented) or `statement_level` (not implemented). Recorded on every result row: the published LEET-Arg numbers use different protocols for different models, so results are only comparable within a protocol.
- `--adapter` -- `hf` (default), `api` (stub, raises `NotImplementedError`), or `stub` (canned outputs, for smoke-testing the harness on a machine with no GPU or torch).
- `--verify-gold-only` -- run gold derivation verification and exit without loading a model.

Model configs are shared with `run_benchmark.py`: per-model JSON under `model_configs/`, named by replacing `/` with `_` in the model id, falling back to the Llama config.

## Dataset

`data/leet_arg/leet_arg_clean_v1.json`, 97 records. Two structural shapes, and the harness branches on both explicitly:

- **85 records with 3 statements.** The five choices are combinations of sub-statements `(a)`, `(b)`, `(c)`.
- **12 records with 5 statements.** Choice *k* simply is statement *k*.

The majority-class answer covers 25/97 records, so **the majority-class baseline is 25.8%**. It is printed alongside every result as a reference line.

## Gold labels

Two priorities, kept separate:

- **P0** -- the gold choice is `int(record["answer"])`. Choice-level accuracy is the headline metric.
- **P1** -- per-statement labels, derived by parsing the `<choices>` block into the set of sub-statements each choice asserts, then combining with the gold answer and the stem polarity. Positive-polarity stems make the gold choice's statements true; negative-polarity stems ("which is **NOT** appropriate") invert the mapping.

Verification runs on every invocation and **currently derives all 97/97 records with no failures**. Any record whose derivation is ambiguous is reported by id and never silently defaulted or skipped.

Stem polarity is detected from the interrogative stem only, with the enumerated statement list cut away first — the word "not" appears inside statement text in many records and would otherwise produce false positives. **Five records have negative-polarity stems**, all in the 5-statement shape: `2021_01`, `2021_03`, `2021_25`, `2023_22`, `2024_05`.

## Metrics

Per-trial rows are written to `results/<timestamp>_<model>.jsonl`, with a `_summary.json` sidecar. Each row carries `record_id, model, decomposition, trial_index, seed, temperature, raw_output, parsed_choice, parse_status, gold_choice, correct`, plus record shape, domain, category, polarity and reasoner.

`parse_status` distinguishes `ok`, `no_choice_found` and `ambiguous`. This distinction is load-bearing: a 1B model often fails to emit a parseable answer at all, and "failed to produce an answer" is a different finding from "produced a wrong answer". The parse failure rate is the direct motivation for the constrained decoding workstream, so it is reported as a first-class number rather than buried in an error log.

Aggregates printed at the end of a run:

- accuracy over parseable responses, and separately over all attempts
- coverage (parseable / total)
- the 25.8% majority-class baseline as a reference line
- consistency: the fraction of records where all trials produced the same answer

**A weak result here is the expected result.** A 1B model is likely to score at or near chance, possibly below the majority baseline, with a substantial parse failure rate. That is what the project's hypothesis predicts and what the later solver work is measured against. The number is reported as it comes out; prompts are not tuned and failed generations are not retried.

## Colab

`notebooks/leet_arg_colab.ipynb` is a thin wrapper: clone the branch, `pip install`, authenticate to Hugging Face, then shell out to `run_leet_arg.py`. There are no Colab-specific code paths inside `leet_arg/`, so moving to Chameleon Cloud is a config change rather than a rewrite.

## Not built here

No solver, no argumentation framework, no DSL work, no constrained decoding, no frontier API calls, and no changes to the patent proof of concept above. Those are separate workstreams that attach at the `Reasoner` and `ModelAdapter` seams.
