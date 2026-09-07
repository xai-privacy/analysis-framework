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
    default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    help="The Hugging Face model ID (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)"
)
args = parser.parse_args()
MODEL_ID = args.model

# 2. Direct output from your previous argument extraction step
claims_dict = {
    "Alice": "The 'Copyright Act' only requires creativity as a condition for a work, and does not demand morality. In order to encourage creation and promote cultural diversity, the recognition of a work should be value-neutral.",
    "Bob": "While the 'Criminal Act' prohibits the production and distribution of obscene materials, protecting the resulting obscene materials as copyrighted works effectively grants rights to those with 'dirty hands' who have committed illegal acts, and recognizes property value and protection for the fruits of illegal conduct, which should not be realized as property rights. This contradicts the principles of legal unity and fairness.",
    "Charlie": "Clearly socially harmful obscene materials—such as child pornography or videos recording actual rape—should not be recognized as works, but for other types of obscene materials, recognizing them as works can help minimize the infringement on freedom of expression and property rights caused by regulation of obscene materials."
}

attacks_list = [
    ('Bob', 'Alice'),
    ('Alice', 'Bob')
]

# 3. Raw verbatim statements passed as input
raw_statements = {
    "statement_1": "(a) Alice presupposes that creativity cannot be acknowledged for obscene forms of expression.",
    "statement_2": "(b) Bob does not regard murals painted in legally prohibited locations or works that incite the public in violation of the National Security Act as objects of protection under the Copyright Act.",
    "statement_3": "(c) Charlie presupposes that even within the same era and region, the legal evaluation of obscenity may vary depending on the purpose, method, and audience of distribution."
}

# 4. Simplified Pydantic schema for evaluation output
class StatementEvaluation(BaseModel):
    is_correct: bool = Field(description="True if the statement's logical analysis is correct, False if it is incorrect")
    reasoning: str = Field(description="Concise logical explanation of why the statement is correct or incorrect")

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
    <statements>
    "{statement_text}"
    </statements>
    
    Task:
    Evaluate if the provided statement is logically correct based on the argument graph.
    1. Set `is_correct` to True if the statement's assessment is logically accurate, or False if it is logically flawed or incorrect.
    2. Provide your `reasoning` for why the statement is correct or incorrect.
    Given the arguments you identified and their attack relations, which of the following statements in <statements> is/are correct as an analysis of the above passage?
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
        return_dict=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Extract response
    prompt_length = inputs["input_ids"].shape[1]
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
    
    status = "CORRECT" if res.is_correct else "INCORRECT"
    print(f"[{stmt_id}]")
    print(f"  Evaluation : {status}")
    print(f"  Reasoning  : {res.reasoning}\n")