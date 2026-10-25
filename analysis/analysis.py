
import pandas as pd

df = pd.read_csv("results/experiment.csv")

# Accuracy for each prompt variant
df["correct"] = df["prediction"] == df["correct_label"]

print("Accuracy by variant:")
print(df.groupby("variant")["correct"].mean())

# Number of distinct predictions for each message
agreement = df.groupby("id")["prediction"].nunique()

print("\nMessages with identical predictions:", (agreement == 1).sum())
print("Messages with differing predictions:", (agreement > 1).sum())

# Show incorrect predictions
print("\nIncorrect classifications:")
print(df.loc[~df["correct"], ["id", "variant", "correct_label", "prediction"]])
