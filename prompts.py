# System prompts for legal causal reasoning evaluation
import os

_BASE_PROMPT = """Solve the problem and provide explanations for your solution.
Select the most appropriate option from <choices> to answer the <question>.
Do not repeat the statements. Answer only in English using this format: Answer-{choice}. Rationale 1. {explanation} 2. {explanation}...

Answer only in JSON format with the following keys: "model_answer" and "model_rationale". The "model_answer" key should contain the selected choice, and the "model_rationale" key should contain a list of explanations for your answer.
"""

def get_system_prompt():
	"""Return the single English prompt used by the benchmark."""
	return _BASE_PROMPT
