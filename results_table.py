import numpy as np
import os

# Define the hyperparameters
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
optimizers = ["adam", "sgd"]
base_dir = "results/segthor/ce"

# Function to get the highest Dice score
def get_highest_dice_score(opt, lr):
    filename = os.path.join(base_dir, f"opt_{opt}", f"lr_{lr}", "dice_val.npy")
    if os.path.exists(filename):
        metrics = np.load(filename)
        return metrics.mean(axis=1).mean(axis=1).max()
    else:
        return None

# Collect the results
results = {}
for opt in optimizers:
    results[opt] = {}
    for lr in learning_rates:
        score = get_highest_dice_score(opt, lr)
        results[opt][lr] = score

# Generate LaTeX table
latex_table = """
\\begin{table}[h!]
\\centering
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
Optimizer & 0.0001 & 0.0005 & 0.001 & 0.005 & 0.01 \\\\
\\hline
"""

for opt in optimizers:
    latex_table += f"{opt.capitalize()} "
    for lr in learning_rates:
        score = results[opt][lr]
        if score is not None:
            latex_table += f"& {score:.4f} "
        else:
            latex_table += "& N/A "
    latex_table += "\\\\\n\\hline\n"

latex_table += """
\\end{tabular}
\\caption{Highest Dice scores for different hyperparameter settings}
\\label{tab:dice_scores}
\\end{table}
"""

# Save the LaTeX table to a file
with open("dice_scores_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved to dice_scores_table.tex")