import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Create the 'Results' folder if it doesn't exist
results_folder = "Results"
os.makedirs(results_folder, exist_ok=True)

# List of conditions to process
conditions = ["baseline", "truncated", "low_res", "truncated_low_res"]

# Iterate through each condition
for condition in conditions:
    print(f"Processing results for condition: {condition}")
    
    # Load results for the current condition
    results_file = f"clip_predictions_{condition}.csv"
    if not os.path.exists(results_file):
        print(f"Results file not found for condition: {condition}")
        continue
    
    results_df = pd.read_csv(results_file)

    # Compute accuracy
    acc_truth = (results_df["truth_pred"] == results_df["truth_gt"]).mean()
    acc_change = (results_df["change_pred"] == results_df["change_gt"]).mean()

    print(f"Quantifier Sentence Accuracy ({condition}): {acc_truth:.2f}")
    print(f"Change Detection Accuracy ({condition}): {acc_change:.2f}")

    # Accuracy split by quantifier
    quantifier_accuracy = results_df.groupby("quantifier").apply(
        lambda group: {
            "truth_accuracy": (group["truth_pred"] == group["truth_gt"]).mean(),
            "change_accuracy": (group["change_pred"] == group["change_gt"]).mean()
        }
    )
    print(f"Accuracy by Quantifier ({condition}):")
    print(quantifier_accuracy)

    # ROC curves for thresholds
    fpr_truth, tpr_truth, thresholds_truth = roc_curve(results_df["truth_gt"], results_df["sim_1"])
    roc_auc_truth = auc(fpr_truth, tpr_truth)

    fpr_change, tpr_change, thresholds_change = roc_curve(results_df["change_gt"], results_df["sim_2"])
    roc_auc_change = auc(fpr_change, tpr_change)

    plt.figure()
    plt.plot(fpr_truth, tpr_truth, label=f"Truth ROC (AUC = {roc_auc_truth:.2f})")
    plt.plot(fpr_change, tpr_change, label=f"Change ROC (AUC = {roc_auc_change:.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({condition})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_folder, f"roc_curves_{condition}.png"))
    plt.close()

    # Cosine similarity distributions
    plt.figure()
    results_df[results_df["truth_gt"] == True]["sim_1"].hist(alpha=0.5, label="True (Truth)")
    results_df[results_df["truth_gt"] == False]["sim_1"].hist(alpha=0.5, label="False (Truth)")
    results_df[results_df["change_gt"] == True]["sim_2"].hist(alpha=0.5, label="Changed (Change)")
    results_df[results_df["change_gt"] == False]["sim_2"].hist(alpha=0.5, label="Unchanged (Change)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(f"Cosine Similarity Distributions ({condition})")
    plt.legend()
    plt.savefig(os.path.join(results_folder, f"cosine_similarity_distributions_{condition}.png"))
    plt.close()

    # Compare performance by quantifier
    plt.figure()
    sns.boxplot(x="quantifier", y="sim_1", data=results_df)
    plt.title(f"Cosine Similarity by Quantifier (Truth) ({condition})")
    plt.xlabel("Quantifier")
    plt.ylabel("Cosine Similarity")
    plt.savefig(os.path.join(results_folder, f"cosine_similarity_by_quantifier_truth_{condition}.png"))
    plt.close()

    plt.figure()
    sns.boxplot(x="quantifier", y="sim_2", data=results_df)
    plt.title(f"Cosine Similarity by Quantifier (Change) ({condition})")
    plt.xlabel("Quantifier")
    plt.ylabel("Cosine Similarity")
    plt.savefig(os.path.join(results_folder, f"cosine_similarity_by_quantifier_change_{condition}.png"))
    plt.close()

    # Compute quantifier verification accuracy by quantifier
    quantifier_accuracy = results_df.groupby("quantifier").apply(
        lambda group: (group["truth_pred"] == group["truth_gt"]).mean()
    )
    print("Quantifier Verification Accuracy:")
    print(quantifier_accuracy)

    # Visualize accuracy by quantifier
    plt.figure()
    sns.barplot(x=quantifier_accuracy.index, y=quantifier_accuracy.values)
    plt.title("Quantifier Verification Accuracy (Each vs Every)")
    plt.xlabel("Quantifier")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(results_folder, "quantifier_verification_accuracy.png"))
    plt.close()

    # Compute change detection accuracy by quantifier and condition
    for condition in conditions:
        results_file = f"clip_predictions_{condition}.csv"
        if not os.path.exists(results_file):
            continue
        results_df = pd.read_csv(results_file)

        change_accuracy = results_df.groupby("quantifier").apply(
            lambda group: (group["change_pred"] == group["change_gt"]).mean()
        )
        print(f"Change Detection Accuracy ({condition}):")
        print(change_accuracy)

        # Visualize change detection accuracy
        plt.figure()
        sns.barplot(x=change_accuracy.index, y=change_accuracy.values)
        plt.title(f"Change Detection Accuracy (Each vs Every) - {condition}")
        plt.xlabel("Quantifier")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(results_folder, f"change_detection_accuracy_{condition}.png"))
        plt.close()

    # Aggregate accuracy across conditions
    accuracy_by_condition = []
    for condition in conditions:
        results_file = f"clip_predictions_{condition}.csv"
        if not os.path.exists(results_file):
            continue
        results_df = pd.read_csv(results_file)

        quantifier_accuracy = results_df.groupby("quantifier").apply(
            lambda group: (group["truth_pred"] == group["truth_gt"]).mean()
        )
        quantifier_accuracy["condition"] = condition
        accuracy_by_condition.append(quantifier_accuracy)

    # Combine results into a single DataFrame
    accuracy_df = pd.concat(accuracy_by_condition).reset_index()

    # Visualize speed-accuracy trade-off
    plt.figure()
    sns.lineplot(data=accuracy_df, x="condition", y=0, hue="quantifier", marker="o")
    plt.title("Speed-Accuracy Trade-off (Each vs Every)")
    plt.xlabel("Condition")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(results_folder, "speed_accuracy_tradeoff.png"))
    plt.close()

    # Visualize similarity score separation
    for condition in conditions:
        results_file = f"clip_predictions_{condition}.csv"
        if not os.path.exists(results_file):
            continue
        results_df = pd.read_csv(results_file)

        plt.figure()
        sns.boxplot(x="truth_gt", y="sim_1", hue="quantifier", data=results_df)
        plt.title(f"Cosine Similarity Separation (Truth) - {condition}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Cosine Similarity")
        plt.savefig(os.path.join(results_folder, f"similarity_separation_truth_{condition}.png"))
        plt.close()

        plt.figure()
        sns.boxplot(x="change_gt", y="sim_2", hue="quantifier", data=results_df)
        plt.title(f"Cosine Similarity Separation (Change) - {condition}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Cosine Similarity")
        plt.savefig(os.path.join(results_folder, f"similarity_separation_change_{condition}.png"))
        plt.close()

    # Analyze error patterns
    for condition in conditions:
        results_file = f"clip_predictions_{condition}.csv"
        if not os.path.exists(results_file):
            continue
        results_df = pd.read_csv(results_file)

        errors = results_df[
            (results_df["quantifier"] == "each") &
            (results_df["truth_gt"] == False) &
            (results_df["truth_pred"] == True)
        ]
        print(f"Error Patterns for Each ({condition}):")
        print(errors)

        # Save error patterns to a CSV for further analysis
        errors.to_csv(os.path.join(results_folder, f"errors_each_{condition}.csv"), index=False)