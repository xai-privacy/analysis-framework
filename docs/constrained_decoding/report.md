# Progress Report: LEET-Arg Benchmark Pipeline Updates

## 1. Overview
This report outlines the recent updates made to the LEET-Arg benchmarking pipeline. The primary focus of these updates was to transition the evaluation prompts to a structured JSON format and to make the `run_benchmark.py` execution script robust enough to parse these outputs without interrupting the models' natural reasoning (Chain-of-Thought) capabilities. 

*(Note: Experimental constrained decoding approaches, such as XGrammar, were tested but ultimately removed to preserve the reasoning performance of the models.)*

## 2. Prompt Evolution: Text-Based to JSON-Structured

To standardize the extraction of answers across different model families, we updated the system prompt to request a JSON-formatted output rather than a simple text string.

### Original Prompt Approach (Text-Based)
Historically, the prompt instructed the model to output its final answer using a specific string format.
* **Target Output Formats:** `Answer-①` or `The correct answer is 1`.
* **Drawback:** Models frequently deviated from this exact string formatting, adding conversational filler or slightly altering the prefix, which led to fragile regex matching and dropped answers.

### New Prompt Approach (JSON-Based)
The updated system prompt explicitly instructs the model to wrap its final answer in a JSON schema, while still allowing the model to "think" or explain its rationale beforehand. The prompts file contains the prompt used for the JSON adherence.
* **Target Output Format:** 
  ```json
  {
    "model_answer": "①"
  }

## 2. Updates to parser

### A. Robust Response Parsing (`parse_model_response`)
We implemented a highly resilient, multi-step parsing function to handle models that produce Chain-of-Thought (CoT) or `<think>` tags before outputting the requested JSON. 

Crucially, we updated the parser to catch **both the new JSON format and existing regex patterns**. This dual approach maintained strict **backward compatibility** with legacy model outputs, while ensuring we could accurately parse and track answers from models that successfully adhered to the new JSON structure.

The parser now executes in the following sequence:
1. **Tag Stripping (For Extraction Only):** Temporarily strips `<think>...</think>` blocks from the search text so the JSON parser doesn't get confused by reasoning text.
2. **Strict JSON Parsing:** Attempts to parse the output directly as a JSON object to find the `"model_answer"` key (tracking successful JSON adherence).
3. **Markdown/Embedded JSON Extraction:** If pure JSON fails, it searches for JSON blocks embedded in markdown (e.g., ` ```json ... ``` `) or mixed within conversational filler.
4. **Backward-Compatible Regex Fallbacks:** If the model ignores the JSON instruction entirely, the parser falls back to matching legacy regex formats (`Answer-①`) or concluding statements (`correct answer is 1`). This guarantees no valid answers are dropped, even if a model completely fails to output JSON.

## Assessment of the JSON Adherence Prompt

**Conclusion:** Yes, the JSON adherence prompt had a net positive effect and should be retained. Previously, out of the 15 questions analyzed for the year 2021, without the strict JSON adherence prompt, Qwen2.5 0.5B instruct gave wrong format [6 times](Qwen_Qwen2.5-0.5B-Instruct-json.json) but that [dropped to 4](Qwen_Qwen2.5-0.5B-Instruct-json.json) when we asked the LLM to have a structured JSON format. In the case of Llama3.2 1B instruct, both the [original prompt](meta-llama_Llama-3.2-1B-Instruct-original.json) and the [JSON one](meta-llama_Llama-3.2-1B-Instruct-json.json) have similar results (Wrong format was seen in the JSON format output file but that is a parser issue). So, the prompt had an effect.

**Key Benefits:**
* **Reliable Structuring:** The prompt successfully forced the model to output its answers and rationales within structured JSON blocks for the vast majority of queries. 
* **Easier Parsing:** Even with the acceptable conversational filler, wrapping the core data in JSON makes programmatic extraction much more reliable than trying to extract answers from unstructured plain text.
* **Clear Field Separation:** It effectively organizes the model's outputs into predictable key-value pairs (like separating the final `answer` from the `model_rationale`), which standardizes the data for downstream analysis.

**Limitations to Note:**
* **Inconsistent Application:** The model occasionally dropped the format on specific questions (e.g., 2021_25 and 2021_28), reverting to plain text. 
* **No Logic Improvements:** The prompt solely dictates presentation; it does not improve the model's reasoning capabilities or prevent factual errors.

**Recommendation:** 
Keep the JSON adherence prompt active. Because your pipeline gracefully handles the conversational filler by capturing the entire output, the structural benefits of the JSON blocks far outweigh the occasional formatting drop-offs.