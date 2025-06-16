import matplotlib.pyplot as plt

from data_loader import *

def generate(data: pd.DataFrame):
    """
    Generate a plot showing the edit distance between the response text and the image text.
    """

    def edit_dist(a, b):
        from Levenshtein import distance
        return distance(a, b)

    counts = Counter()
    for _, row in data.iterrows():
        gt = str(row["image"])
        pred = str(row["response_text"])
        d = edit_dist(gt, pred)
        counts[d] += 1

    distances = sorted(counts.items())
    xs, ys = zip(*distances)
    plt.bar(xs, ys)
    plt.xlabel("Edit Distance")
    plt.ylabel("Number of CAPTCHAs")
    plt.title("Distribution of Edit Distances (Model vs Ground Truth)")
    plt.xticks(range(15))
    plt.savefig(f"../plots/edit_distance_histogram_{model}.png")
    plt.show()

if __name__ == '__main__':
    model = model_4  # Change this to the desired model
    print(f"Generating edit distance histogram for {model}")
    generate(get_data(model, depth=1))