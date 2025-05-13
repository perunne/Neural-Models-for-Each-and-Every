import os
import pandas as pd
from collections import Counter
from itertools import product

# Path to your metadata
METADATA_PATH = "stimuli/metadata.csv"

# Load metadata
df = pd.read_csv(METADATA_PATH)

if not os.path.exists(METADATA_PATH):
    print(f"Metadata file not found at: {METADATA_PATH}")
    print("Please check the file path and try again.")
    exit(1)

# 1. Simple counts #
print("== Global Counts ==")
print(df.shape[0], "total trials")
print(df['quantifier'].value_counts(), "\n")
print(df['color_label'].value_counts(), "\n")
print(df['is_true'].value_counts().rename({1: "True", 0: "False"}), "\n")
print(df['is_change'].value_counts().rename({1: "Changed", 0: "Unchanged"}), "\n")

# 2. Cross-tabulation #
print("== Condition Combinations ==")
combo_counts = df.groupby(['quantifier', 'color_label', 'is_true', 'is_change']).size()
print(combo_counts)

# Print as table
print("\n== Tabulated by Quantifier and Truth ==")
print(pd.crosstab([df['quantifier'], df['color_label']], [df['is_true'], df['is_change']]))

# 3. Check for missing or uneven combinations #
print("\n== Missing Combinations ==")
quantifiers = df['quantifier'].unique()
colors = df['color_label'].unique()
truth_vals = [0, 1]
change_vals = [0, 1]

expected_combos = list(product(quantifiers, colors, truth_vals, change_vals))
existing_combos = set(combo_counts.index)

missing = [combo for combo in expected_combos if combo not in existing_combos]
if missing:
    print("Missing combinations:")
    for m in missing:
        print(f"  Quantifier: {m[0]}, Color: {m[1]}, is_true: {m[2]}, is_change: {m[3]}")
else:
    print("All good.")
