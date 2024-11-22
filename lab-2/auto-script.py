import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm

k_values = range(1, 11)
normalize_methods = ["none", "normalize", "min-max"]
accuracies = {method: [] for method in normalize_methods}

for method in tqdm(normalize_methods, desc="Methods"):
    for k in tqdm(k_values, desc="k-values", leave=False):
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
                accuracy = float(accuracy_str)
                accuracies[method].append(accuracy)
                break
        else:
            accuracies[method].append(0.0)  # Append 0.0 if accuracy not found

# Plot the accuracies
for method in normalize_methods:
    plt.plot(k_values, accuracies[method], label=method)

plt.xlabel("k")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs k for different normalization methods")
plt.legend()
plt.show()
