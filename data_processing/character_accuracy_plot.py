from data_loader import *

def generate(data: pd.DataFrame):
    """
    Generate a plot showing the per-character accuracy of CAPTCHA responses.
    This accounts for character frequency in the ground truth.
    """
    # -------------- 1. Setup containers -----------------
    chars = get_chars(1)
    char_hits = {c: 0 for c in chars}  # correct predictions
    char_total = {c: 0 for c in chars}  # ground-truth appearances

    # -------------- 2. Tally hits & totals ---------------
    for gt, pred in zip(data["image"].astype(str), data["response_text"].astype(str)):
        gt = gt
        pred = pred

        # skip badly aligned pairs (e.g. wrong length) - should never happen as we use cleaned data
        if len(gt) != len(pred):
            print(f"Skipping misaligned pair: GT='{gt}', Pred='{pred}'")
            continue

        for g_ch, p_ch in zip(gt, pred):
            if g_ch in chars:  # ignore stray symbols
                char_total[g_ch] += 1
                if g_ch == p_ch:
                    char_hits[g_ch] += 1

    # -------------- 3. Build arrays for plotting ----------
    labels = []
    freq = []  # x-axis: how many times char appeared
    accuracy = []  # y-axis: (hits / total)

    for c in chars:
        if char_total[c] == 0:  # ignore chars that never appear
            print(f"Warning: Character '{c}' never appeared in ground truth.")
            continue
        labels.append(c)
        freq.append(char_total[c])
        accuracy.append(char_hits[c] / char_total[c])

    freq = np.array(freq)
    accuracy = np.array(accuracy)

    # -------------- 4. Scatter plot -----------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(freq, accuracy, color="steelblue")

    # annotate each point with its character label
    for x, y, lbl in zip(freq, accuracy, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points",
                     xytext=(5, 3), fontsize=8)

    plt.xlabel("Ground-truth occurrences (frequency)")
    plt.ylabel("Per-character accuracy")
    plt.title("Character Accuracy vs. Frequency")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"../plots/character_accuracy_plot_{model}.png")
    plt.show()

if __name__ == '__main__':
    model = model_4  # Change this to the desired model
    print(f"Generating character accuracy plot for {model}")
    generate(get_cleaned_data(model, depth=1))
