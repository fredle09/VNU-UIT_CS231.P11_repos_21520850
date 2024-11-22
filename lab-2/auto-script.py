import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

k_values = range(1, 11)
normalize_methods = ["none", "normalize", "min-max"]
accuracies = {method: [0.0] * len(k_values) for method in normalize_methods}


def run_command(method, k):
    cmd = [
        "python",
        "main.py",
        "--stage",
        "all",
        "-k",
        str(k),
        "--use-normalize",
        method,
        "--log",
        "INFO",
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if "Accuracy:" in line:
            accuracy_str = line.split("Accuracy:")[-1].strip().replace("%", "")
            return float(accuracy_str)
    return 0.0  # Return 0.0 if accuracy not found


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for method in normalize_methods:
            for idx, k in enumerate(k_values):
                futures.append(executor.submit(run_command, method, k))
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Running commands",
        ):
            accuracy = future.result()
            method = normalize_methods[futures.index(future) // len(k_values)]
            idx = futures.index(future) % len(k_values)
            accuracies[method][idx] = accuracy

    # Plot the accuracies
    for method in normalize_methods:
        plt.plot(k_values, accuracies[method], label=method)

    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs k for different normalization methods")
    plt.legend()
    plt.show()
