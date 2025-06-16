import string

from data_loader import *

def generate(data: pd.DataFrame):
    # ---------------------------------------------------------------- 1. Alphabet
    cols = list(string.ascii_letters + string.digits)              # X-axis: A–Z + a–z  (52)
    col_map = {c: i for i, c in enumerate(cols)}

    rows = [c for c in get_chars(1)]        # Y-axis from user fn (case-insensitive)
    rows = sorted(set(rows))                       # ensure unique & sorted
    if not rows:
        raise ValueError("get_chars() returned an empty list.")

    row_map = {c: i for i, c in enumerate(rows)}

    conf = np.zeros((len(rows), len(cols)), dtype=int)   # shape  (|rows| × 52)

    # --------------------------------------------------------- 2. Fill confusion
    for gt, pred in zip(data["image"].astype(str), data["response_text"].astype(str)):
        if len(gt) != len(pred):
            continue
        for g, p in zip(gt, pred):               # GT → upper; pred keep case
            if g in row_map and p in col_map:
                conf[row_map[g], col_map[p]] += 1

    # ---------------------------------------------------- 3. Row-normalise (%)
    row_sum = conf.sum(axis=1, keepdims=True)
    conf_pct = conf / np.maximum(row_sum, 1) * 100

    # Mask diagonal—must map TRUE char (row) to SAME char column if present
    diag_mask = np.zeros_like(conf_pct, dtype=bool)
    for r, c_true in enumerate(rows):
        # match both upper and lower versions in columns
        for col_variant in c_true:
            if col_variant in col_map:
                diag_mask[r, col_map[col_variant]] = True

    conf_masked = np.ma.masked_where(diag_mask, conf_pct)

    # ----------------------------------------------------------- 4. Plot
    plt.figure(figsize=(13, max(4, 0.25 * len(rows))))
    im = plt.imshow(conf_masked, cmap="Reds", aspect="auto")

    plt.xticks(np.arange(len(cols)), cols, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(rows)), rows)
    plt.xlabel("Predicted character (ASCII letters)")
    plt.ylabel("True character (from get_chars())")
    plt.title("Character-level Confusion Matrix (% of appearances)")

    cbar = plt.colorbar(im)
    cbar.set_label("% of true-char occurrences\npredicted as column-char")

    plt.tight_layout()
    plt.savefig(f"../plots/confusion_matrix_heatmap_{model}.png")
    plt.show()

if __name__ == '__main__':
    model = model_4  # Change this to the desired model
    print(f"Generating confusion matrix heatmap for {model}")
    generate(get_cleaned_data(model, depth=1))

