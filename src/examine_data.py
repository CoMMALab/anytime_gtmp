import pickle
from pathlib import Path

def load_motion_planning_dataset(
    robot_name="panda",
    base_data_dir=None,
    verbose=False
):
    if base_data_dir is None:
        base_data_dir = Path.cwd() / "data"
    else:
        base_data_dir = Path(base_data_dir)

    data_dir = base_data_dir / robot_name
    problems_file = data_dir / "problems.pkl"

    if not problems_file.exists():
        raise FileNotFoundError(f"Dataset not found at: {problems_file}")

    with open(problems_file, "rb") as f:
        data = pickle.load(f)

    problems = data.get("problems", {})
    if not problems:
        raise ValueError("No problems found in the dataset!")

    if verbose:
        print(f"Found {len(problems)} problem categories:")
        for problem_name, problem_list in problems.items():
            print(f"\n📂 Problem Category: {problem_name}")
            for i, prob in enumerate(problem_list):
                index = prob.get("index", i)
                start = prob.get("start", [])
                goals = prob.get("goals", [])
                valid = prob.get("valid", False)

                start_shape = (len(start),) if isinstance(start, list) else getattr(start, "shape", "N/A")
                goals_shape = (
                    (len(goals), len(goals[0]) if goals and isinstance(goals[0], list) else "N/A")
                    if isinstance(goals, list) else getattr(goals, "shape", "N/A")
                )

                print(f"  - [Index {index}]")
                print(f"      Valid      : {valid} (type: {type(valid).__name__})")
                print(f"      Start      : shape {start_shape}, type: {type(start).__name__}")
                print(f"      Goals      : shape {goals_shape}, type: {type(goals).__name__}")

    return data

def main():
    try:
        # Load dataset (adjust robot name or path if needed)
        dataset = load_motion_planning_dataset(robot_name="panda", base_data_dir=Path.cwd() / "data", verbose=True)

        # Basic structure checks
        problems = dataset.get("problems", {})
        if not problems:
            print("No motion planning problems found.")
            return

        print("\nDataset loaded successfully.")
        print(f"Number of problem categories: {len(problems)}")

        # Print sample entry
        for category_name, entries in problems.items():
            print(f"\n🧪 Sample from category: {category_name}")
            if entries:
                sample = entries[0]
                print("  → Sample Problem:")
                print(f"     Index: {sample.get('index')}")
                print(f"     Valid: {sample.get('valid')}")
                print(f"     Start: {sample.get('start')}")
                print(f"     Goals: {sample.get('goals')}")
            break  # Only print one category
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
