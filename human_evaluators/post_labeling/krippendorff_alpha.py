import json
from pathlib import Path

import pandas as pd
import krippendorff

# One JSON export per labeler. The filename, without .json, is that person's
# name in the grid. Each file can come from its own Label Studio project.
EXPORT_FILENAMES = [
    "ann1.json",
    "ann2.json",
]

# these are each question that labelers need to fill out in the labeling
# interface. The loop below builds one agreement table per name. 
QUESTIONS = [
    "answer_correctness",
    "reasoning_quality",
    "stmt_a_eval",
    "stmt_b_eval",
    "stmt_c_eval",
]

def load_exports(paths):
    # Tag every task with the file it came from. A solo Label Studio project
    # numbers its only user as 1, so completed_by cannot tell two exports apart.
    labeled = []
    for path in paths:
        with open(path) as f:
            exported = json.load(f)
        labeler = Path(path).stem
        for task in exported:
            labeled.append((labeler, task))
    return labeled

def choice(annotation, from_name):
    # Each labeling's "result" is one object per filled-in control:
    # "result": [
    #   {
    #     "from_name": "answer_correctness",
    #     "type": "choices",
    #     "value": { "choices": ["Correct (Matches Benchmark)"] }
    #   },
    #   {
    #     "from_name": "reasoning_quality",
    #     "type": "choices",
    #     "value": { "choices": ["Valid"] }
    #   }
    # ]
    # The string we keep is value.choices[0].
    for item in annotation.get("result", []):
        if item.get("from_name") == from_name and item.get("type") == "choices":
            choices = item.get("value", {}).get("choices") or []
            return choices[0] if choices else None
    return None

def agreement_alphas(labeled_tasks):
    # Return {from_name: alpha} for each choice question.
    # labeled_tasks is a list of (labeler name, task) pairs from load_exports.
    alphas = {}
    for QUESTION in QUESTIONS:
        rows = []
        for labeler, task in labeled_tasks:
            data = task["data"]
            # The export also contains the labeling interface imported as a task.
            # That task's data is the XML and has no id, model, or condition.
            if not all(key in data for key in ("id", "model", "condition")):
                continue
            # One column per generated answer. id is the LEET ARG question (2021_18),
            # model is the LLM that wrote the response (Model 1), and condition is the
            # prompting setup (Baseline, CoT, or Argumentation Framework).
            unit = f"{data['id']}|{data['model']}|{data['condition']}"
            # task["annotations"] is this file's submissions for this task.
            # The labeler name is the export filename, not completed_by.
            # This whole pass runs
            # once for each QUESTION string, which is one from_name. choice()
            # finds that from_name in the labeling's result list and returns
            # value.choices[0], the string the labeler picked. None means this
            # labeling has no answer for that question, so it is skipped.
            # Each appended row is one cell of this question's grid: the
            # labeler, the model output, and that choice string.
            for ann in task.get("annotations", []):
                if ann.get("was_cancelled"):
                    continue
                label = choice(ann, QUESTION)
                if label is None:
                    continue
                rows.append({
                    "annotator": labeler,
                    "unit": unit,
                    "label": label,
                    "updated_at": ann.get("updated_at", ""),
                })

        # If the same labeler submitted the same model output more than once,
        # sort oldest to newest and keep only the latest choice.
        df = (
            pd.DataFrame(rows)
            .sort_values("updated_at")
            .drop_duplicates(["annotator", "unit"], keep="last")
        )
        # Spread the long list into the coder-by-item grid. Rows are labelers,
        # columns are model outputs, and the cell is the choice string. A
        # labeler who never rated that output has no row for it, so the cell
        # is NaN.
        # e.g.
        #   annotator     2021_18|Model 1|CoT   2021_20|Model 3|Argumentation Framework
        #   evaluator_a   Correct               Correct
        #   evaluator_b   NaN                   Correct
        wide = df.pivot(index="annotator", columns="unit", values="label")
        print(QUESTION)
        print(wide)

        try:
            alpha = float(krippendorff.alpha(wide.to_numpy().astype(str), level_of_measurement="nominal"))
        except ValueError:
            # Every label in this table is the same, so chance agreement
            # cannot be estimated and alpha is undefined.
            print("undefined: every label is the same")
            alphas[QUESTION] = None
            continue
        print(alpha)
        alphas[QUESTION] = alpha
    return alphas

if __name__ == "__main__":
    agreement_alphas(load_exports(EXPORT_FILENAMES))
