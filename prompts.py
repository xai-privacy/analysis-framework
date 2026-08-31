# System prompts for legal causal reasoning evaluation
import os

_BASE_PROMPT = """Solve the problem and provide explanations for your solution.
Select the most appropriate option from <choices> to answer the <question>.
Do not repeat the statements. Answer only in English using this format: Answer-{choice}. Rationale 1. {explanation} 2. {explanation}...
"""

# Answer strictly in valid JSON format containing exactly two keys:
# 1. "model_answer": The choice you selected (e.g., "①", "②").
# 2. "model_rationale": A detailed string explaining your reasoning.
# CRITICAL: You must escape all double quotes inside your JSON string values using a backslash (e.g., \"), or use single quotes for internal quotations.

def get_system_prompt():
	"""Return the single English prompt used by the benchmark."""
	return _BASE_PROMPT
