from pathlib import Path
import numpy as np
import pandas as pd

data = pd.DataFrame(columns=["image", "response_text", "accuracy"])

with open("results.txt", "r") as f:
    counter = 0
    for line in f:
        if counter >= 300: # Limit to 300 entries
            break
        image, response_text, accuracy = line.strip().split(";")
        data.loc[len(data)] = [image, response_text, float(accuracy)]
        counter += 1

print(data)
prev_len = len(data)

# Remove data with "INVALID" in response_text
data = data[~data["response_text"].str.contains("INVALID", na=False)]

print(data)
invalid_amount = prev_len - len(data)
print(f"Removed {invalid_amount} entries with 'INVALID' in response_text.")

mean_accuracy = data["accuracy"].mean()
print(f"Mean accuracy: {mean_accuracy:.2f}")

confidence_interval = 1.96 * (data["accuracy"].std() / np.sqrt(len(data)))
print(f"95% Confidence interval: {mean_accuracy - confidence_interval:.2f} to {mean_accuracy + confidence_interval:.2f}")

perfect_scores = data[data["accuracy"] == 1.0]
print(f"Perfect scores: {len(perfect_scores)}")
