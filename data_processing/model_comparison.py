from Levenshtein import distance
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from data_loader import *


def comparison_mcnemar(model_4: pd.DataFrame, model_27: pd.DataFrame):
    """
    Perform McNemar's test to compare the performance of two models.
    Returns the p-value of the test.
    """

    # Create a contingency table
    contingency_table = pd.crosstab(
        model_4["response_text"] == model_4["image"],
        model_27["response_text"] == model_27["image"]
    )

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=False, correction=True)

    return result.statistic, result.pvalue

def comparison_edit_distance(data_1: pd.DataFrame, data_2: pd.DataFrame):
    """
    Calculate the average edit distance between the response text and the image text for two models.
    Returns a tuple of average edit distances for both models.
    """
    # -- 1. Levenshtein distance for each row -----------------------------------
    d1 = [distance(gt, pred) for gt, pred in
          zip(data_1["image"].astype(str), data_1["response_text"].astype(str))]
    d2 = [distance(gt, pred) for gt, pred in
           zip(data_2["image"].astype(str), data_2["response_text"].astype(str))]

    d1 = np.array(d1, dtype=int)
    d2 = np.array(d2, dtype=int)

    # -- 2. Summary stats --------------------------------------------------------
    mean1, mean2 = d1.mean(), d2.mean()
    delta = mean2 - mean1

    # -- 3. Paired Wilcoxon test -------------------------------------------------
    stat, p = wilcoxon(d2, d1)

    # -- 4. Output ---------------------------------------------------------------
    return mean1, mean2, delta, stat, p

def generate_wilson(model: pd.DataFrame):
    """
    Calculate the Wilson score interval for the accuracy of a model.
    Returns the lower and upper bounds of the interval.
    """
    n = len(model)
    p = model["response_text"].eq(model["image"]).mean()
    z = 1.96  # 95% confidence level

    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

    return lower_bound, upper_bound

def generate(data_1: pd.DataFrame, data_2: pd.DataFrame):
    """ Generate a comparison statistics for different models based on their accuracy."""
    # -- 1. Calculate Wilson score intervals for both models --------------------
    mcnemar_result = comparison_mcnemar(data_1, data_2)
    print(f"McNemar's test statistic: {mcnemar_result[0]}, p-value: {mcnemar_result[1]}")

    # -- 2. Calculate Wilson score intervals for both models --------------------
    edit_distance_result = comparison_edit_distance(data_1, data_2)
    print(f"Edit-distance comparison (lower = better)")
    print(f"  Data-1  mean = {edit_distance_result[0]:.2f}")
    print(f"  Data-2  mean = {edit_distance_result[1]:.2f}")
    print(f"  Δ (D1 − D2)   = {edit_distance_result[2]:+.2f}")
    print(f"  Wilcoxon  W = {edit_distance_result[3]:.0f},  p = {edit_distance_result[4]:.4g}")

    # -- 3. Calculate Wilson score intervals for both models --------------------
    wilson_4 = generate_wilson(data_1)
    wilson_27 = generate_wilson(data_2)
    print(f"Wilson score intervals:")
    print(f"  Data-1  = [{wilson_4[0]:.4f}, {wilson_4[1]:.4f}]")
    print(f"  Data-2  = [{wilson_27[0]:.4f}, {wilson_27[1]:.4f}]")

    captcha_completions_1 = data_1["response_text"].eq(data_1["image"]).sum()
    captcha_completions_2 = data_2["response_text"].eq(data_2["image"]).sum()
    print(f"Captcha completions:")
    print(f"  Data-1  completions = {captcha_completions_1}")
    print(f"  Data-2  completions = {captcha_completions_2}")

    captcha_completion_ratio_1 = (captcha_completions_1 / len(data_1))
    captcha_completion_ratio_2 = (captcha_completions_2 / len(data_2))
    print(f"Captcha completion ratio:")
    print(f"  Data-1  completion ratio = {captcha_completion_ratio_1:.4f}")
    print(f"  Data-2  completion ratio = {captcha_completion_ratio_2:.4f}")

def generate_prompts():
    data1 = get_data_raw(model_27, prompt=1, depth=1)
    data2 = get_data_raw(model_27, prompt=2, depth=1)
    data3 = get_data_raw(model_27, prompt=3, depth=1)
    print(f"Lengths of datasets: Data-1 = {len(data1)}, Data-2 = {len(data2)}, Data-3 = {len(data3)}")
    print("=" * 50)
    print("Generating comparison statistics for parameter 1 and 2")
    generate(data1, data2)
    print("=" * 50)
    print("Generating comparison statistics for parameter 2 and 3")
    generate(data2, data3)
    print("=" * 50)
    print("Generating comparison statistics for parameter 1 and 3")
    generate(data1, data3)
    print("=" * 50)

if __name__ == '__main__':
    data_1 = get_data(model_4, depth=1)
    data_2 = get_data(model_27, depth=1)
    print(f"Lengths of datasets: Data-1 = {len(data_1)}, Data-2 = {len(data_2)}")
    generate(data_1, data_2)




