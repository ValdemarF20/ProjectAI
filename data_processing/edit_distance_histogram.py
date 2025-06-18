import matplotlib.pyplot as plt

from data_loader import *

def generate(data: pd.DataFrame, model: str, prompt: int = -1):
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
    prompt_str = f"_{prompt}" if prompt != -1 else ""
    plt.savefig(f"../plots/edit_distance_histogram_{model}{prompt_str}.png")
    plt.show()

def generate_parameters(model: str):
    data1 = get_data(model, prompt=1, depth=1)
    data2 = get_data(model, prompt=2, depth=1)
    data3 = get_data(model, prompt=3, depth=1)
    print(f"Generating edit distance histogram for {model} with prompt 1")
    generate(data1, model, 1)
    print(f"Generating edit distance histogram for {model} with prompt 2")
    generate(data2, model, 2)
    print(f"Generating edit distance histogram for {model} with prompt 3")
    generate(data3, model, 3)

def generate_all(model: str):
    print(f"Generating edit distance histogram for {model}")
    generate(get_data(model, depth=1), model)

if __name__ == '__main__':
    generate_parameters(model_27)