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

def comparison_edit_distance(model_4: pd.DataFrame, model_27: pd.DataFrame):
    """
    Calculate the average edit distance between the response text and the image text for two models.
    Returns a tuple of average edit distances for both models.
    """
    # -- 1. Levenshtein distance for each row -----------------------------------
    d4 = [distance(gt, pred) for gt, pred in
          zip(model_4["image"].astype(str), model_4["response_text"].astype(str))]
    d27 = [distance(gt, pred) for gt, pred in
           zip(model_27["image"].astype(str), model_27["response_text"].astype(str))]

    d4 = np.array(d4, dtype=int)
    d27 = np.array(d27, dtype=int)

    # -- 2. Summary stats --------------------------------------------------------
    mean4, mean27 = d4.mean(), d27.mean()
    delta = mean27 - mean4

    # -- 3. Paired Wilcoxon test -------------------------------------------------
    stat, p = wilcoxon(d27, d4)

    # -- 4. Output ---------------------------------------------------------------
    return mean4, mean27, delta, stat, p

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

def generate(model_4: pd.DataFrame, model_27: pd.DataFrame):
    """ Generate a comparison statistics for different models based on their accuracy."""
    # -- 1. Calculate Wilson score intervals for both models --------------------
    mcnemar_result = comparison_mcnemar(model_4, model_27)
    print(f"McNemar's test statistic: {mcnemar_result[0]}, p-value: {mcnemar_result[1]}")

    # -- 2. Calculate Wilson score intervals for both models --------------------
    edit_distance_result = comparison_edit_distance(model_4, model_27)
    print(f"Edit-distance comparison (lower = better)")
    print(f"  Model-4B  mean = {edit_distance_result[0]:.2f}")
    print(f"  Model-27B mean = {edit_distance_result[1]:.2f}")
    print(f"  Δ (27B − 4B)   = {edit_distance_result[2]:+.2f}")
    print(f"  Wilcoxon  W = {edit_distance_result[3]:.0f},  p = {edit_distance_result[4]:.4g}")

    # -- 3. Calculate Wilson score intervals for both models --------------------
    wilson_4 = generate_wilson(model_4)
    wilson_27 = generate_wilson(model_27)
    print(f"Wilson score intervals:")
    print(f"  Model-4B  = [{wilson_4[0]:.4f}, {wilson_4[1]:.4f}]")
    print(f"  Model-27B = [{wilson_27[0]:.4f}, {wilson_27[1]:.4f}]")


if __name__ == '__main__':
    model_4_data = get_data(model_4, depth=1)
    model_27_data = get_data(model_27, depth=1)
    print(f"Lengths of datasets: Model-4B = {len(model_4_data)}, Model-27B = {len(model_27_data)}")
    generate(get_data(model_4, depth=1), get_data(model_27, depth=1))




