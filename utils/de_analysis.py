import numpy as np
import pandas as pd
from scipy import stats


def run_differential_expression(
    df: pd.DataFrame,
    control_cols: list,
    treated_cols: list,
    pval_threshold: float = 0.05,
    fc_threshold: float = 1.5,
    already_log2: bool = False,
) -> pd.DataFrame:
    """
    Perform differential expression analysis using Welch's t-test + BH FDR correction.

    Args:
        df: Gene expression matrix (genes x samples) — raw counts or pre-normalized
        control_cols: Column names for control samples
        treated_cols: Column names for treated samples
        pval_threshold: Adjusted p-value cutoff
        fc_threshold: Fold change threshold (linear)
        already_log2: Set True for microarray / TPM / FPKM data that is already
                      log2-transformed or normalized. Raw RNA-seq counts should use False.

    Returns:
        DataFrame with log2FC, p-value, adj. p-value, significance label
    """
    if already_log2:
        log2_control = df[control_cols].values.astype(float)
        log2_treated = df[treated_cols].values.astype(float)
    else:
        # Pseudocount to avoid log(0)
        pseudo = 1.0
        control_data = df[control_cols].values.astype(float) + pseudo
        treated_data = df[treated_cols].values.astype(float) + pseudo
        log2_control = np.log2(control_data)
        log2_treated = np.log2(treated_data)

    # Mean log2 expression per group
    mean_control = log2_control.mean(axis=1)
    mean_treated = log2_treated.mean(axis=1)

    log2_fc = mean_treated - mean_control

    # Welch's t-test (unequal variance)
    t_stats, p_values = stats.ttest_ind(log2_treated, log2_control, axis=1, equal_var=False)
    p_values = np.nan_to_num(p_values, nan=1.0)

    # Benjamini-Hochberg FDR correction
    adj_p = _bh_correction(p_values)

    # Mean expression (average of both groups)
    mean_expr = (mean_control + mean_treated) / 2

    results = pd.DataFrame({
        "Gene": df.index,
        "log2FoldChange": log2_fc,
        "pvalue": p_values,
        "padj": adj_p,
        "meanExpression": mean_expr,
        "mean_control": mean_control,
        "mean_treated": mean_treated,
    })

    # Significance labels
    log2_fc_thresh = np.log2(fc_threshold)
    results["significance"] = "Not significant"
    results.loc[
        (results["padj"] < pval_threshold) & (results["log2FoldChange"] > log2_fc_thresh),
        "significance"
    ] = "Upregulated"
    results.loc[
        (results["padj"] < pval_threshold) & (results["log2FoldChange"] < -log2_fc_thresh),
        "significance"
    ] = "Downregulated"

    results["-log10pval"] = -np.log10(results["pvalue"].clip(1e-300))
    results = results.sort_values("padj").reset_index(drop=True)
    return results


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adj = p_values * n / ranks
    # Enforce monotonicity
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return np.clip(adj, 0, 1)


def get_top_degs(results: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N differentially expressed genes by adjusted p-value."""
    sig = results[results["significance"] != "Not significant"]
    return sig.head(n)


def get_summary_stats(results: pd.DataFrame) -> dict:
    """Return a summary dict of DE results."""
    return {
        "total_genes": len(results),
        "upregulated": (results["significance"] == "Upregulated").sum(),
        "downregulated": (results["significance"] == "Downregulated").sum(),
        "not_significant": (results["significance"] == "Not significant").sum(),
    }
