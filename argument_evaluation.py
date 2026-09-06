import argparse
import json
import re
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Parse Command Line Arguments
parser = argparse.ArgumentParser(description="Run argument evaluation with a specific Hugging Face model.")
parser.add_argument(
    "--model", 
    type=str, 
    default="meta-llama/Meta-Llama-3.2-1B-Instruct", 
    help="The Hugging Face model ID (e.g., meta-llama/Meta-Llama-3.2-1B-Instruct)"
)
args = parser.parse_args()
MODEL_ID = args.model

# 2. Direct output from your previous argument extraction step
claims_dict = {
    'Alice': "If the appointment of Supreme Court justices is left exclusively to the president and the sovereign people have no means of control, there is no way to restrain a justice no matter how biased their way of thinking may be. We need to introduce this system to establish a mechanism for judicial oversight by the people.",
    'Bob': "That’s a reasonable point. However, the method used in Country X has the problem that unless voters proactively express their desire for removal, it is treated as opposition to removal, which means it does not properly reflect the will of the voters. If this system is introduced exactly as it is, it could soon become meaningless.",
    'Charlie': "If improvements are made, those concerns can be dispelled. But ultimately, if this system is introduced, Supreme Court justices will be swayed by popular opinion instead of judging cases according to law and their beliefs, and the independence of the judiciary will be undermined."
}

attacks_list = [
    ('Bob', 'Alice'),
    ('Charlie', 'Alice'),
    ('Charlie', 'Bob')
]

# 3. Raw verbatim statements passed as input
raw_statements = {
    "statement_1": "① If Country Y’s constitution guarantees lifetime tenure for Supreme Court justices appointed by the president (except in the case of irreversible physical disability), Alice’s opinion is strengthened.",
    "statement_2": "② If in Country Y, public opinion surveys indicate endless criticisms such as \"justice depends on wealth\" regarding court decisions, and public trust in the judiciary has declined annually, Alice’s opinion is strengthened.",
    "statement_3": "③ If, in Country X, no Supreme Court justice has been dismissed through public review in the past 70 years and only around 10% of total votes cast each time favored dismissal, Bob’s opinion is weakened.",
    "statement_4": "④ If in Country Y, some Supreme Court justices have caused repeated social chaos by overturning established Supreme Court decisions purely to pursue popularity, this fact strengthens Charlie’s opinion.",
    "statement_5": "⑤ If in Country Y, information related to each justice’s rulings is not properly provided and media coverage mainly focuses on personal activities for entertainment value, making it difficult for sound public opinion regarding confidence in Supreme Court justices to be formed, this strengthens Charlie’s opinion."
}

# 4. Pydantic schema for evaluation output
class StatementEvaluation(BaseModel):
    target_speaker: str = Field(description="Speaker evaluated in the statement ('Alice', 'Bob', or 'Charlie')")
    claimed_effect: str = Field(description="The effect claimed by the statement ('strengthens' or 'weakens')")
    actual_effect: str = Field(description="The true logical impact of the scenario on the speaker's claim ('strengthens' or 'weakens')")
    is_appropriate: bool = Field(description="True if claimed_effect matches actual_effect; False if it misrepresents the impact")
    reasoning: str = Field(description="Concise logical explanation of why the scenario strengthens or weakens the claim")

# Initialize Hugging Face Model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device
)

def evaluate_verbatim_statement(statement_text: str, claims: dict, attacks: list) -> StatementEvaluation:
    system_prompt = f"""You are a formal logic evaluator analyzing a debate context.
Respond ONLY with a valid JSON object. Do not include any markdown formatting, conversational text, or explanations outside the JSON.
Your JSON must strictly adhere to this schema:
{json.dumps(StatementEvaluation.model_json_schema(), indent=2)}
"""

    user_prompt = f"""
    Argument Graph Claims:
    {json.dumps(claims, indent=2)}
    
    Attack Relations (Attacker -> Target):
    {attacks}
    
    Verbatim Evaluation Statement to Analyze:
    "{statement_text}"
    
    Task:
    1. Identify which speaker is targeted and whether the statement claims their position 'strengthens' or 'weakens'.
    2. Determine whether the hypothetical condition in the statement actually STRENGTHENS or WEAKENS that speaker's argument.
    3. Set `is_appropriate` to True if the statement's claimed effect matches the actual logical impact, or False if the statement is incorrect.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Format prompt and generate
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True  # Explicitly request a dictionary to fix the AttributeError
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,         # Unpack the dictionary (inputs_ids and attention_mask)
            max_new_tokens=512,
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Extract response
    prompt_length = inputs["input_ids"].shape[1] # Reference the specific tensor for the shape
    response_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
    
    # Isolate JSON in case the model hallucinates markdown tags
    json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
        
    return StatementEvaluation.model_validate_json(response_text)

# 5. Run evaluation loop
print("\n--- Verbatim Statement Analysis Results ---\n")

for stmt_id, stmt_text in raw_statements.items():
    res = evaluate_verbatim_statement(stmt_text, claims_dict, attacks_list)
    
    status = "APPROPRIATE" if res.is_appropriate else "NOT APPROPRIATE (INCORRECT)"
    print(f"[{stmt_id}]")
    print(f"  Target Speaker : {res.target_speaker}")
    print(f"  Claimed Effect : {res.claimed_effect}")
    print(f"  Actual Impact  : {res.actual_effect}")
    print(f"  Evaluation     : {status}")
    print(f"  Reasoning      : {res.reasoning}\n")