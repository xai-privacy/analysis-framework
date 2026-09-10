import json
import random

def create_evaluation_packages(
    input_file_path="LEET_Arg_Questions_Test_Set.json",
    output_file_path="Evaluator_Work_Assignment.json",
    seed=42
):
    random.seed(seed)

    # 1. Load Question IDs from input JSON
    with open(input_file_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    question_ids = [q["id"] for q in questions_data[:16]]  # 16 questions

    models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]
    conditions = ["Baseline", "CoT", "Argumentation Framework"]
    evaluators = ["Evaluator A", "Evaluator B", "Evaluator C", "Evaluator D", "Evaluator E", "Evaluator F"]

    # 2. Build the 240 unique outputs (16 questions * 5 models * 3 conditions)
    all_outputs = []
    for q_id in question_ids:
        for m in models:
            for c in conditions:
                all_outputs.append({
                    "question_id": q_id,
                    "model": m,
                    "condition": c
                })

    # 3. Repeat assignment until a perfectly balanced allocation is found
    success = False
    while not success:
        evaluator_workloads = {e: [] for e in evaluators}
        evaluator_counts = {e: 0 for e in evaluators}
        valid = True

        for item in all_outputs:
            # Build an array of three distinct evaluators to be
            # assigned a certain output item
            # Evaluators who still have capacity (< 120 tasks)
            available = [e for e in evaluators if evaluator_counts[e] < 120]

            if len(available) < 3:
                valid = False
                break

            assigned = random.sample(available, 3)
            for ev in assigned:
                evaluator_workloads[ev].append(item)
                evaluator_counts[ev] += 1

        if valid and all(count == 120 for count in evaluator_counts.values()):
            success = True

    # 4. Format output JSON
    output_data = {}
    for ev in evaluators:
        tasks = evaluator_workloads[ev]
        random.shuffle(tasks)  # Randomize task order for the evaluator
        
        output_data[ev] = {
            "total_assigned": len(tasks),  # Exactly 120
            "tasks": tasks
        }

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully generated assignments in '{output_file_path}'.")
    for ev, count in evaluator_counts.items():
        print(f"{ev}: {count} tasks assigned")

if __name__ == "__main__":
    create_evaluation_packages(
        input_file_path="LEET_Arg_Questions_Test_Set.json",
        output_file_path="Evaluator_Work_Assignment.json"
    )