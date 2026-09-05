# System prompts for legal causal reasoning evaluation
import os

_BASE_PROMPT = """You extract an argument graph from the <Debate>.

Return ONLY one valid JSON object with exactly these keys:

{
  "claims": {
    "SpeakerID": "one concise string containing that speaker's core position"
  },
  "attacks": [
    ["SourceSpeakerID", "TargetSpeakerID"]
  ]
}

Rules for "claims":
- Use the speaker names exactly as they appear in the debate, such as "Alice", "Bob", or "Charlie".
- Map each speaker to exactly one string.
- The string must summarize that speaker's main position, not a quotation, question, option, or rationale.
- Do not use nested dictionaries or lists as claim values.

Rules for "attacks":
- Each attack must be a two-item list: ["source", "target"].
- Both items must be speaker IDs appearing as keys in "claims".
- ["Alice", "Bob"] means Alice's argument attacks, contradicts, or directly challenges Bob's argument.
- Include an attack only when the debate explicitly contains a rebuttal or contradiction.
- Do not use statement numbers, choice numbers, claim text, "Attacker", or invented speaker names.
- Do not create attacks merely because speakers discuss the same topic.
- Do not create self-attacks.
- If there are no explicit attacks, return an empty list.

Important:
- Output JSON only.
- Do not wrap the JSON in Markdown fences.
- Use valid JSON double quotes.
- Do not include any explanation or multiple-choice answer.

Example output shape:
{
  "claims": {
    "Alice": "Alice argues that obscene materials should qualify as copyrighted works whenever they satisfy the creativity requirement.",
    "Bob": "Bob argues that illegal obscene materials should not receive copyright protection because doing so would protect the fruits of illegal conduct.",
    "Charlie": "Charlie argues that only socially harmful obscene materials should be excluded from copyright protection."
  },
  "attacks": [
    ["Bob", "Alice"],
    ["Charlie", "Bob"]
  ]
}
"""

### Preserve the prompts we were using previously.

# _BASE_PROMPT = """Solve the problem and provide explanations for your solution.
# Select the most appropriate option from <choices> to answer the <question>.
# Do not repeat the statements. Answer only in English using this format: Answer-{choice}. Rationale 1. {explanation} 2. {explanation}...
# """

# Answer strictly in valid JSON format containing exactly two keys:
# 1. "model_answer": The choice you selected (e.g., "①", "②").
# 2. "model_rationale": A detailed string explaining your reasoning.
# CRITICAL: You must escape all double quotes inside your JSON string values using a backslash (e.g., \"), or use single quotes for internal quotations.

def get_system_prompt():
	"""Return the single English prompt used by the benchmark."""
	return _BASE_PROMPT
