import os, subprocess

# Task 1: Run generation with different temperatures
models = [
    # {"name": "base", "model": "Qwen/Qwen2.5-Math-1.5B"},
    {"name": "instruct", "model": "Qwen/Qwen2.5-Math-1.5B-Instruct"},
]

experiments = [
    {"dataset": "math", "rollout-n": 16, "temperature": 0.6},
    #{"dataset": "math", "rollout-n": 16, "temperature": 1.0},
    #{"dataset": "math", "rollout-n": 16, "temperature": 1.2},
    # {"dataset": "amc", "rollout-n": 64, "temperature": 0.6},
    # {"dataset": "amc", "rollout-n": 64, "temperature": 1.0},
    # {"dataset": "amc", "rollout-n": 64, "temperature": 1.2},
    # {"dataset": "aime", "rollout-n": 64, "temperature": 0.6},
    {"dataset": "aime", "rollout-n": 64, "temperature": 1.0},
    # {"dataset": "aime", "rollout-n": 64, "temperature": 1.2},
]

for model in models:
    for exp in experiments:
        output_dir = f"outputs"
        os.makedirs(output_dir, exist_ok=True)

        inference_cmd = [
            "python",
            "inference.py",
            f"--model={model['model']}",
            f"--dataset={exp['dataset']}",
            f"--temperature={exp['temperature']}",
            f"--rollout-n={exp['rollout-n']}",
            f"--output_file={output_dir}/{model['name']}-{exp['dataset']}-{exp['temperature']}.jsonl",
            f"--batch-size=16",
            f"--dp-size=1",
        ]
        subprocess.run(inference_cmd)
