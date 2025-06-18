import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api_receiver import *


def get_data(model: str, prompt: int = -1, depth: int = 0) -> pd.DataFrame:
    """
    Load data from a specific model and prompt.
    """
    data = pd.DataFrame(columns=["image", "response_text", "accuracy"])
    # Set path depending on depth by adding ".." to the path
    path_fixer = "../" * depth

    if prompt == -1:
        # Load data for all prompts
        for p in range(1, 4):
            with open(f"{path_fixer}prompt_{p}_results/results_{get_model_type(model)}.txt", "r") as f:
                for line in f:
                    if len(line.strip().split(";")) != 3:
                        print(f"Skipping malformed line: {line.strip()}")
                        continue
                    image, response_text, accuracy = line.strip().split(";")
                    data.loc[len(data)] = [image, response_text, float(accuracy)]
    else: # Load data for a specific prompt
        with open(f"{path_fixer}prompt_{prompt}_results/results_{get_model_type(model)}.txt", "r") as f:
            for line in f:
                image, response_text, accuracy = line.strip().split(";")
                data.loc[len(data)] = [image, response_text, float(accuracy)]

    return data

def get_cleaned_data(model: str, prompt: int = -1, depth: int = 0) -> pd.DataFrame:
    """
    Load and clean data from a specific model and prompt.
    Removes entries with "INVALID" in response_text.
    """
    data = get_data(model, prompt, depth)

    # Remove rows with "INVALID" in response_text
    data = data[~data["response_text"].str.contains("INVALID", na=False)]

    return data.reset_index(drop=True)

def get_data_raw(model: str, prompt: int = -1, depth: int = 0) -> pd.DataFrame:
    """
    Load raw data from a specific model and prompt.
    Removes "-INVALID" from the response_text but does not filter out entries.
    """
    data = get_data(model, prompt, depth)
    # Remove all "-INVALID" from the entries but do not remove the entry itself
    data["response_text"] = data["response_text"].str.replace("-INVALID", "", regex=False)
    return data

def analyze_data(data: pd.DataFrame):
    prev_len = len(data)
    print(f"Initial data length: {prev_len}")

    # Remove data with "INVALID" in response_text
    data = data[~data["response_text"].str.contains("INVALID", na=False)]

    invalid_amount = prev_len - len(data)
    print(f"Removed {invalid_amount} invalid entries.")
    invalid_percentage = invalid_amount / prev_len * 100
    print(f"Invalid entries percentage: {invalid_percentage:.2f}%")

    mean_accuracy = data["accuracy"].mean()
    print(f"Mean accuracy: {mean_accuracy:.2f}")

    confidence_interval = 1.96 * (data["accuracy"].std() / np.sqrt(len(data)))
    print(f"95% Confidence interval: {mean_accuracy - confidence_interval:.2f} to {mean_accuracy + confidence_interval:.2f}")

    perfect_scores = data[data["accuracy"] == 1.0]
    print(f"Perfect scores: {len(perfect_scores)}")

    perfect_score_percentage = (len(perfect_scores) / len(data)) * 100
    print(f"Perfect score percentage: {perfect_score_percentage:.2f}%")

def calculate_all_data():
    for model in models:
        print(f"Data for model: {get_model_type(model)}")
        for prompt in range(1, 4):
            print(f"Prompt {prompt}:")
            analyze_data(get_data(model, prompt))
            print("\n" + "="*50 + "\n")

def generate_char_heatmap(data: pd.DataFrame):
    # Allowed CAPTCHA characters (A–Z + 0–9)
    chars = list(string.ascii_letters + string.digits)
    char_to_index = {c: i for i, c in enumerate(chars)}
    confusion = np.zeros((len(chars), len(chars)), dtype=int)

    # Fill confusion matrix
    for _, row in data.iterrows():
        gt = row["image"]
        pred = str(row["response_text"])

        if len(gt) != len(pred):
            continue  # skip misaligned pairs

        for g_char, p_char in zip(gt, pred):
            if g_char in char_to_index and p_char in char_to_index:
                g_idx = char_to_index[g_char]
                p_idx = char_to_index[p_char]
                confusion[g_idx, p_idx] += 1

    # Normalize rows to get accuracy per true character
    row_sums = confusion.sum(axis=1, keepdims=True)
    heatmap = confusion / np.maximum(row_sums, 1)  # avoid division by zero

    # Plot
    plt.figure(figsize=(12, 10))
    im = plt.imshow(heatmap, cmap="Blues")

    plt.xticks(ticks=np.arange(len(chars)), labels=chars, rotation=90)
    plt.yticks(ticks=np.arange(len(chars)), labels=chars)
    plt.xlabel("Predicted character")
    plt.ylabel("True character")
    plt.title("Character Confusion Heatmap (Accuracy per class)")

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Accuracy")

    plt.tight_layout()
    plt.show()

def print_extra_info(data: pd.DataFrame):
    """
    Print extra information about the data.
    """
    chars = list(string.ascii_letters + string.digits)
    # Remove chars from get_chars to return a list of characters that never appear in the data
    non_occurring_chars = chars.copy()
    for c in chars:
        if c in get_chars():
            non_occurring_chars.remove(c)

    print(f"Characters that never appear in the data ({len(non_occurring_chars)}): {', '.join(non_occurring_chars)}")

def get_chars(depth: int = 0) -> list:
    """
    Get a list of characters that appear in the data (not all do).
    """
    path_fixer = "../" * depth

    # Loop through all images in data/samples and collect unique characters
    chars = set()
    for image in Path(f"{path_fixer}data/samples").iterdir():
        if image.is_file() and image.suffix == ".png":
            for char in image.stem:
                if char in string.ascii_letters + string.digits:
                    chars.add(char)
    return sorted(chars)  # Return sorted list of characters

def get_non_occurring_chars(depth: int = 0) -> list:
    """
    Get a list of characters that never appear in the data.
    """
    all_chars = set(string.ascii_letters + string.digits)
    occurring_chars = set(get_chars(depth))
    non_occurring_chars = all_chars - occurring_chars
    return sorted(non_occurring_chars)


if __name__ == '__main__':
    model = model_27  # Change this to the desired model
    print(f"Analyzing data for model: {get_model_type(model)}")
    get_chars()
    #print_extra_info(get_data(model))
    #analyze_data(get_data(model))

