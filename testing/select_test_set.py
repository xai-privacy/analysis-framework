import argparse
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def normalize_category(c):
    if not c:
        return "Unknown"
    s = str(c).lower().strip()
    if "rebuttal" in s:
        return "Argumentation and Rebuttal"
    elif "problem" in s:
        return "Argument Evaluation & Problem Solving"
    elif "analysis" in s:
        if "eval" in s:
            return "Argument Evaluation & Analysis"
        else:
            return "Argument Analysis"
    return str(c).strip()


def normalize_domain(d):
    if d is None or str(d).lower().strip() in ["null", "none", "", "nan"]:
        return "null"
    s = str(d).strip().lower()
    if s == "norms":
        return "Norms"
    elif s == "humanities":
        return "Humanities"
    elif s == "society":
        return "Society"
    return "null"


def select_stratified_sample(input_filepath: str, output_filepath: str, seed: int = 42):
    np.random.seed(seed)

    # Load questions
    with open(input_filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Target counts
    cat_targets = {
        "Argumentation and Rebuttal": 7,
        "Argument Analysis": 3,
        "Argument Evaluation & Problem Solving": 2,
        "Argument Evaluation & Analysis": 4,
    }

    dom_targets = {
        "Norms": 4,
        "Humanities": 7,
        "Society": 3,
        "null": 2,
    }

    N = len(questions)
    dom_keys = list(dom_targets.keys())
    
    # Variables: N binary selection variables + 2 * len(dom_keys) slack variables
    num_vars = N + 2 * len(dom_keys)

    # Random tie-breaking weights for selected items
    r = np.random.uniform(0, 1, N)
    c_obj = np.zeros(num_vars)
    c_obj[:N] = r

    # High penalty costs for domain slack variables
    for j in range(len(dom_keys)):
        c_obj[N + j] = 1000.0  # positive slack
        c_obj[N + len(dom_keys) + j] = 1000.0  # negative slack

    constraints = []

    # Constraint 1: Exactly 16 items total
    row_total = np.zeros(num_vars)
    row_total[:N] = 1
    constraints.append(LinearConstraint(row_total, 16, 16))

    # Constraint 2: Hard Category targets
    for cat, target in cat_targets.items():
        row_cat = np.zeros(num_vars)
        for i, q in enumerate(questions):
            if normalize_category(q.get("category")) == cat:
                row_cat[i] = 1
        constraints.append(LinearConstraint(row_cat, target, target))

    # Constraint 3: Soft Domain targets with slack
    for j, (dom, target) in enumerate(dom_targets.items()):
        row_dom = np.zeros(num_vars)
        for i, q in enumerate(questions):
            if normalize_domain(q.get("domain")) == dom:
                row_dom[i] = 1
        row_dom[N + j] = -1
        row_dom[N + len(dom_keys) + j] = 1
        constraints.append(LinearConstraint(row_dom, target, target))

    # Variable bounds and integrality constraints
    integrality = np.zeros(num_vars)
    integrality[:N] = 1  # Decision variables x_i are binary

    bounds = Bounds(
        lb=np.zeros(num_vars),
        ub=np.concatenate([np.ones(N), np.full(2 * len(dom_keys), np.inf)])
    )

    # Run Mixed-Integer Linear Programming solver
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

    if res.status != 0:
        raise RuntimeError(f"MILP solver failed to find a valid sample: {res.message}")

    selected_indices = np.where(res.x[:N] > 0.5)[0]

    # Structure output JSON items
    output_data = []
    for idx in selected_indices:
        q = questions[idx]
        output_data.append({
            "id": q.get("id"),
            "domain": normalize_domain(q.get("domain")),
            "category": normalize_category(q.get("category")),
        })

    # Export to destination path
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully created stratified sample with {len(output_data)} items at '{output_filepath}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a multi-way stratified sample for LEET-Arg.")
    parser.add_argument(
        "--input",
        default="LEET_Arg_Questions_cleaned_and_rationale_by_statement.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        default="LEET_Arg_Questions_Test_Set.json",
        help="Path to output JSON file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    select_stratified_sample(args.input, args.output, seed=args.seed)