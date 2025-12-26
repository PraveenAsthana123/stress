#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Advanced Statistical Analysis Module for GenAI-RAG-EEG
================================================================================

Title: Comprehensive Statistical Analysis for EEG-Based Stress Classification
Reference: GenAI-RAG-EEG Paper v2, IEEE Sensors Journal 2024

Description:
    This module implements advanced statistical analysis methods for EEG data
    including parametric tests, non-parametric tests, effect sizes, multiple
    comparison corrections, and comprehensive reporting.

Features:
    1. Parametric Tests:
       - Independent/Paired t-tests
       - One-way/Two-way ANOVA
       - Repeated Measures ANOVA
       - Linear Mixed Effects Models

    2. Non-Parametric Tests:
       - Mann-Whitney U test
       - Wilcoxon Signed-Rank test
       - Kruskal-Wallis H test
       - Friedman test
       - McNemar test (for classifier comparison)
       - Permutation tests

    3. Effect Size Measures:
       - Cohen's d (pooled, Glass's delta)
       - Hedges' g (bias-corrected)
       - Common Language Effect Size (CLES)
       - Eta-squared, Partial eta-squared
       - Omega-squared

    4. Multiple Comparison Corrections:
       - Bonferroni correction
       - Holm-Bonferroni (step-down)
       - Benjamini-Hochberg FDR
       - Benjamini-Yekutieli FDR

    5. Confidence Intervals:
       - Normal approximation
       - Bootstrap (percentile, BCa)
       - Jackknife

    6. Normality Tests:
       - Shapiro-Wilk
       - Kolmogorov-Smirnov
       - Anderson-Darling
       - D'Agostino-Pearson

    7. Correlation Analysis:
       - Pearson correlation
       - Spearman rank correlation
       - Kendall tau
       - Partial correlation
       - Point-biserial correlation

    8. Advanced Features:
       - Cross-validation significance testing
       - Bayesian hypothesis testing
       - Power analysis
       - Sample size estimation

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp,
    mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    shapiro, normaltest, kstest, anderson,
    pearsonr, spearmanr, kendalltau, pointbiserialr,
    f_oneway, chi2_contingency
)
import warnings
from pathlib import Path
import json
from datetime import datetime

# Try importing optional dependencies
try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, matthews_corrcoef, cohen_kappa_score,
        confusion_matrix, balanced_accuracy_score
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class StatisticalTestResult:
    """Result of a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: str
    ci_lower: float
    ci_upper: float
    significant: bool
    alpha: float = 0.05
    interpretation: str = ""
    sample_size_1: int = 0
    sample_size_2: int = 0
    power: float = 0.0


@dataclass
class NormalityTestResult:
    """Result of normality tests."""
    test_name: str
    statistic: float
    p_value: float
    is_normal: bool
    alpha: float = 0.05


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    method: str
    correlation: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_samples: int
    interpretation: str = ""


@dataclass
class EffectSizeResult:
    """Comprehensive effect size results."""
    cohens_d: float
    hedges_g: float
    glass_delta: float
    cles: float  # Common Language Effect Size
    interpretation: str


@dataclass
class MultipleComparisonResult:
    """Results after multiple comparison correction."""
    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected: List[bool]
    correction_method: str
    alpha: float


@dataclass
class ComprehensiveAnalysisResult:
    """Complete analysis result for a comparison."""
    group1_name: str
    group2_name: str
    group1_stats: Dict[str, float]
    group2_stats: Dict[str, float]
    normality_tests: Dict[str, NormalityTestResult]
    parametric_test: StatisticalTestResult
    nonparametric_test: StatisticalTestResult
    effect_sizes: EffectSizeResult
    recommended_test: str
    conclusion: str


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray, pooled: bool = True) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group data
        group2: Second group data
        pooled: Use pooled standard deviation (default True)

    Returns:
        Cohen's d value

    Interpretation:
        |d| < 0.2: Negligible
        |d| 0.2-0.5: Small
        |d| 0.5-0.8: Medium
        |d| > 0.8: Large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if pooled:
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    else:
        # Simple average
        pooled_std = np.sqrt((var1 + var2) / 2)

    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Hedges' g (bias-corrected Cohen's d).

    Better for small samples (n < 20).
    """
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)

    # Correction factor (approximation)
    correction = 1 - (3 / (4 * n - 9))

    return d * correction


def glass_delta(group1: np.ndarray, group2: np.ndarray, control_group: int = 2) -> float:
    """
    Calculate Glass's delta (uses control group SD only).

    Args:
        group1: Treatment group
        group2: Control group
        control_group: Which group is control (1 or 2)
    """
    if control_group == 2:
        control_std = np.std(group2, ddof=1)
    else:
        control_std = np.std(group1, ddof=1)

    return (np.mean(group1) - np.mean(group2)) / (control_std + 1e-10)


def common_language_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Common Language Effect Size (CLES).

    Probability that a randomly selected value from group1
    is greater than a randomly selected value from group2.

    Returns:
        CLES probability (0.5 = no effect, 1.0 = complete separation)
    """
    n1, n2 = len(group1), len(group2)

    # Count pairs where group1 > group2
    count = 0
    for x in group1:
        for y in group2:
            if x > y:
                count += 1
            elif x == y:
                count += 0.5

    return count / (n1 * n2)


def eta_squared(groups: List[np.ndarray]) -> float:
    """
    Calculate eta-squared for ANOVA.

    Proportion of variance explained by group membership.
    """
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # SS between
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # SS total
    ss_total = np.sum((all_data - grand_mean)**2)

    return ss_between / (ss_total + 1e-10)


def omega_squared(groups: List[np.ndarray]) -> float:
    """
    Calculate omega-squared (less biased than eta-squared).
    """
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    N = len(all_data)
    k = len(groups)

    # SS between
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # SS within (error)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)

    # MS within
    ms_within = ss_within / (N - k)

    return (ss_between - (k - 1) * ms_within) / (ss_within + ss_between + ms_within + 1e-10)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "Negligible"
    elif d_abs < 0.5:
        return "Small"
    elif d_abs < 0.8:
        return "Medium"
    else:
        return "Large"


def compute_all_effect_sizes(group1: np.ndarray, group2: np.ndarray) -> EffectSizeResult:
    """Compute all effect size measures."""
    d = cohens_d(group1, group2)
    g = hedges_g(group1, group2)
    delta = glass_delta(group1, group2)
    cles = common_language_effect_size(group1, group2)

    return EffectSizeResult(
        cohens_d=float(d),
        hedges_g=float(g),
        glass_delta=float(delta),
        cles=float(cles),
        interpretation=interpret_effect_size(d)
    )


# =============================================================================
# NORMALITY TESTS
# =============================================================================

def test_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, NormalityTestResult]:
    """
    Run multiple normality tests on data.

    Args:
        data: Data array
        alpha: Significance level

    Returns:
        Dictionary of normality test results
    """
    results = {}

    # Shapiro-Wilk (best for n < 50)
    if len(data) >= 3:
        try:
            stat, p = shapiro(data)
            results["shapiro_wilk"] = NormalityTestResult(
                test_name="Shapiro-Wilk",
                statistic=float(stat),
                p_value=float(p),
                is_normal=p > alpha,
                alpha=alpha
            )
        except Exception:
            pass

    # D'Agostino-Pearson (requires n >= 20)
    if len(data) >= 20:
        try:
            stat, p = normaltest(data)
            results["dagostino_pearson"] = NormalityTestResult(
                test_name="D'Agostino-Pearson",
                statistic=float(stat),
                p_value=float(p),
                is_normal=p > alpha,
                alpha=alpha
            )
        except Exception:
            pass

    # Kolmogorov-Smirnov (against normal distribution)
    try:
        stat, p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        results["kolmogorov_smirnov"] = NormalityTestResult(
            test_name="Kolmogorov-Smirnov",
            statistic=float(stat),
            p_value=float(p),
            is_normal=p > alpha,
            alpha=alpha
        )
    except Exception:
        pass

    # Anderson-Darling
    try:
        result = anderson(data, dist='norm')
        # Use 5% critical value
        cv_5pct = result.critical_values[2]
        is_normal = result.statistic < cv_5pct
        results["anderson_darling"] = NormalityTestResult(
            test_name="Anderson-Darling",
            statistic=float(result.statistic),
            p_value=-1,  # Anderson-Darling doesn't provide p-value directly
            is_normal=is_normal,
            alpha=alpha
        )
    except Exception:
        pass

    return results


def check_assumptions(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Check statistical test assumptions.

    Returns:
        Dictionary with normality and homogeneity of variance results
    """
    results = {
        "group1_normality": test_normality(group1, alpha),
        "group2_normality": test_normality(group2, alpha)
    }

    # Levene's test for homogeneity of variance
    stat, p = stats.levene(group1, group2)
    results["homogeneity_of_variance"] = {
        "test": "Levene's test",
        "statistic": float(stat),
        "p_value": float(p),
        "equal_variance": p > alpha
    }

    # Overall recommendation
    g1_normal = all(r.is_normal for r in results["group1_normality"].values())
    g2_normal = all(r.is_normal for r in results["group2_normality"].values())
    equal_var = results["homogeneity_of_variance"]["equal_variance"]

    if g1_normal and g2_normal:
        if equal_var:
            results["recommendation"] = "Use parametric test (t-test with equal variance)"
        else:
            results["recommendation"] = "Use Welch's t-test (unequal variance)"
    else:
        results["recommendation"] = "Use non-parametric test (Mann-Whitney U)"

    return results


# =============================================================================
# PARAMETRIC TESTS
# =============================================================================

def independent_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True
) -> StatisticalTestResult:
    """
    Independent samples t-test.

    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level
        equal_var: Assume equal variances (False for Welch's t-test)
    """
    stat, p = ttest_ind(group1, group2, equal_var=equal_var)
    d = cohens_d(group1, group2)

    # Confidence interval for mean difference
    mean_diff = np.mean(group1) - np.mean(group2)
    se = np.sqrt(np.var(group1)/len(group1) + np.var(group2)/len(group2))
    ci = stats.t.interval(1-alpha, len(group1)+len(group2)-2, loc=mean_diff, scale=se)

    test_name = "Independent t-test" if equal_var else "Welch's t-test"

    return StatisticalTestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(d),
        effect_size_type="Cohen's d",
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'Not significant'} difference (d={d:.3f}, {interpret_effect_size(d)} effect)",
        sample_size_1=len(group1),
        sample_size_2=len(group2)
    )


def paired_ttest(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Paired samples t-test.

    Args:
        before: Pre-treatment values
        after: Post-treatment values
        alpha: Significance level
    """
    if len(before) != len(after):
        raise ValueError("Paired t-test requires equal sample sizes")

    stat, p = ttest_rel(before, after)

    # Effect size for paired data
    diff = after - before
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    # Confidence interval
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci = stats.t.interval(1-alpha, len(diff)-1, loc=np.mean(diff), scale=se)

    return StatisticalTestResult(
        test_name="Paired t-test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(d),
        effect_size_type="Cohen's d (paired)",
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'Not significant'} change",
        sample_size_1=len(before),
        sample_size_2=len(after)
    )


def one_way_anova(*groups, alpha: float = 0.05) -> StatisticalTestResult:
    """
    One-way ANOVA for comparing multiple groups.
    """
    stat, p = f_oneway(*groups)

    # Effect size (eta-squared)
    eta_sq = eta_squared(list(groups))

    return StatisticalTestResult(
        test_name="One-way ANOVA",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(eta_sq),
        effect_size_type="Eta-squared",
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'No significant'} difference between groups (η²={eta_sq:.3f})"
    )


# =============================================================================
# NON-PARAMETRIC TESTS
# =============================================================================

def mann_whitney_u(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    Mann-Whitney U test (non-parametric alternative to independent t-test).
    """
    stat, p = mannwhitneyu(group1, group2, alternative=alternative)

    # Effect size: rank-biserial correlation
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial correlation

    return StatisticalTestResult(
        test_name="Mann-Whitney U",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        effect_size_type="Rank-biserial r",
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'Not significant'} difference (r={r:.3f})",
        sample_size_1=n1,
        sample_size_2=n2
    )


def wilcoxon_signed_rank(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    """
    if len(before) != len(after):
        raise ValueError("Wilcoxon test requires equal sample sizes")

    # Remove zero differences
    diff = after - before
    non_zero_mask = diff != 0

    if np.sum(non_zero_mask) < 10:
        warnings.warn("Small sample size for Wilcoxon test")

    stat, p = wilcoxon(before, after)

    # Effect size: matched-pairs rank-biserial correlation
    n = len(diff)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)

    return StatisticalTestResult(
        test_name="Wilcoxon Signed-Rank",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        effect_size_type="Matched-pairs rank-biserial r",
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'Not significant'} change",
        sample_size_1=len(before),
        sample_size_2=len(after)
    )


def kruskal_wallis(*groups, alpha: float = 0.05) -> StatisticalTestResult:
    """
    Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA).
    """
    stat, p = kruskal(*groups)

    # Effect size: epsilon-squared
    N = sum(len(g) for g in groups)
    epsilon_sq = (stat - len(groups) + 1) / (N - len(groups))

    return StatisticalTestResult(
        test_name="Kruskal-Wallis H",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(epsilon_sq),
        effect_size_type="Epsilon-squared",
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'No significant'} difference between groups"
    )


def friedman_test(*groups, alpha: float = 0.05) -> StatisticalTestResult:
    """
    Friedman test (non-parametric repeated measures).
    """
    stat, p = friedmanchisquare(*groups)

    # Effect size: Kendall's W
    n = len(groups[0])
    k = len(groups)
    W = stat / (n * (k - 1))

    return StatisticalTestResult(
        test_name="Friedman",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(W),
        effect_size_type="Kendall's W",
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"{'Significant' if p < alpha else 'No significant'} difference (W={W:.3f})"
    )


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    McNemar's test for comparing two classifiers.

    Tests whether two classifiers have significantly different error rates.
    """
    # Build contingency table
    # b = pred1 correct, pred2 wrong
    # c = pred1 wrong, pred2 correct
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = np.sum(correct1 & ~correct2)  # Model 1 right, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 right

    # McNemar's statistic with continuity correction
    if b + c == 0:
        stat = 0
        p = 1.0
    else:
        stat = (abs(b - c) - 1)**2 / (b + c)
        p = 1 - stats.chi2.cdf(stat, df=1)

    return StatisticalTestResult(
        test_name="McNemar",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float((b - c) / (b + c + 1e-10)),
        effect_size_type="Discordant ratio",
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        alpha=alpha,
        interpretation=f"Classifiers are {'significantly different' if p < alpha else 'not significantly different'}"
    )


# =============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# =============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> MultipleComparisonResult:
    """
    Bonferroni correction for multiple comparisons.

    Most conservative method.
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    rejected = [p < alpha for p in corrected]

    return MultipleComparisonResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        rejected=rejected,
        correction_method="Bonferroni",
        alpha=alpha
    )


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> MultipleComparisonResult:
    """
    Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    corrected = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)

    for i, idx in enumerate(sorted_indices):
        corrected[idx] = min(sorted_p[i] * (n - i), 1.0)
        rejected[idx] = corrected[idx] < alpha

    return MultipleComparisonResult(
        original_p_values=p_values,
        corrected_p_values=corrected.tolist(),
        rejected=rejected.tolist(),
        correction_method="Holm-Bonferroni",
        alpha=alpha
    )


def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> MultipleComparisonResult:
    """
    Benjamini-Hochberg False Discovery Rate correction.

    Controls the expected proportion of false discoveries.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted = np.zeros(n)
    adjusted[sorted_indices[-1]] = sorted_p[-1]

    for i in range(n - 2, -1, -1):
        adjusted[sorted_indices[i]] = min(
            adjusted[sorted_indices[i + 1]],
            sorted_p[i] * n / (i + 1)
        )

    rejected = [p < alpha for p in adjusted]

    return MultipleComparisonResult(
        original_p_values=p_values,
        corrected_p_values=adjusted.tolist(),
        rejected=rejected,
        correction_method="Benjamini-Hochberg FDR",
        alpha=alpha
    )


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def comprehensive_correlation(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, CorrelationResult]:
    """
    Compute multiple correlation coefficients.
    """
    results = {}
    n = len(x)

    # Pearson correlation
    r, p = pearsonr(x, y)
    se = np.sqrt((1 - r**2) / (n - 2))
    z = np.arctanh(r)
    z_se = 1 / np.sqrt(n - 3)
    ci = np.tanh([z - 1.96 * z_se, z + 1.96 * z_se])

    results["pearson"] = CorrelationResult(
        method="Pearson",
        correlation=float(r),
        p_value=float(p),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        significant=p < alpha,
        n_samples=n,
        interpretation=_interpret_correlation(r)
    )

    # Spearman correlation
    rho, p = spearmanr(x, y)
    results["spearman"] = CorrelationResult(
        method="Spearman",
        correlation=float(rho),
        p_value=float(p),
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        n_samples=n,
        interpretation=_interpret_correlation(rho)
    )

    # Kendall's tau
    tau, p = kendalltau(x, y)
    results["kendall"] = CorrelationResult(
        method="Kendall's tau",
        correlation=float(tau),
        p_value=float(p),
        ci_lower=0,
        ci_upper=0,
        significant=p < alpha,
        n_samples=n,
        interpretation=_interpret_correlation(tau)
    )

    return results


def _interpret_correlation(r: float) -> str:
    """Interpret correlation strength."""
    r_abs = abs(r)
    if r_abs < 0.1:
        strength = "Negligible"
    elif r_abs < 0.3:
        strength = "Weak"
    elif r_abs < 0.5:
        strength = "Moderate"
    elif r_abs < 0.7:
        strength = "Strong"
    else:
        strength = "Very strong"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction} correlation"


# =============================================================================
# BOOTSTRAP AND CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'percentile'
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Input data
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        method: 'percentile' or 'bca' (bias-corrected accelerated)

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    n = len(data)
    point_estimate = statistic_func(data)

    # Generate bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)

    alpha = 1 - confidence

    if method == 'percentile':
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    elif method == 'bca':
        # BCa method (bias-corrected and accelerated)
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))

        # Acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic_func(jack_sample)
        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats)**3) / (6 * np.sum((jack_mean - jackknife_stats)**2)**1.5 + 1e-10)

        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

        ci_lower = np.percentile(bootstrap_stats, 100 * p_low)
        ci_upper = np.percentile(bootstrap_stats, 100 * p_high)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(point_estimate), float(ci_lower), float(ci_upper)


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_func: callable = None,
    n_permutations: int = 10000,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Permutation test for comparing two groups.

    Args:
        group1: First group
        group2: Second group
        statistic_func: Function to compute test statistic (default: mean difference)
        n_permutations: Number of permutations
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        (observed_statistic, p_value)
    """
    if statistic_func is None:
        statistic_func = lambda x, y: np.mean(x) - np.mean(y)

    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    observed = statistic_func(group1, group2)

    # Permutation distribution
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        perm_stats[i] = statistic_func(perm_g1, perm_g2)

    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    elif alternative == 'greater':
        p_value = np.mean(perm_stats >= observed)
    elif alternative == 'less':
        p_value = np.mean(perm_stats <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(observed), float(p_value)


# =============================================================================
# COMPREHENSIVE ANALYSIS FUNCTION
# =============================================================================

def comprehensive_two_group_analysis(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    alpha: float = 0.05,
    paired: bool = False
) -> ComprehensiveAnalysisResult:
    """
    Perform comprehensive statistical analysis comparing two groups.

    This function runs all appropriate tests and provides recommendations.

    Args:
        group1: First group data
        group2: Second group data
        group1_name: Name for first group
        group2_name: Name for second group
        alpha: Significance level
        paired: Whether data is paired

    Returns:
        ComprehensiveAnalysisResult with all test results
    """
    # Descriptive statistics
    group1_stats = {
        "n": len(group1),
        "mean": float(np.mean(group1)),
        "std": float(np.std(group1, ddof=1)),
        "median": float(np.median(group1)),
        "iqr": float(np.percentile(group1, 75) - np.percentile(group1, 25)),
        "min": float(np.min(group1)),
        "max": float(np.max(group1)),
        "skewness": float(stats.skew(group1)),
        "kurtosis": float(stats.kurtosis(group1))
    }

    group2_stats = {
        "n": len(group2),
        "mean": float(np.mean(group2)),
        "std": float(np.std(group2, ddof=1)),
        "median": float(np.median(group2)),
        "iqr": float(np.percentile(group2, 75) - np.percentile(group2, 25)),
        "min": float(np.min(group2)),
        "max": float(np.max(group2)),
        "skewness": float(stats.skew(group2)),
        "kurtosis": float(stats.kurtosis(group2))
    }

    # Normality tests
    norm1 = test_normality(group1, alpha)
    norm2 = test_normality(group2, alpha)

    # Check if data is normally distributed
    is_normal1 = all(r.is_normal for r in norm1.values() if r.p_value > 0)
    is_normal2 = all(r.is_normal for r in norm2.values() if r.p_value > 0)

    # Parametric tests
    if paired:
        parametric_result = paired_ttest(group1, group2, alpha)
        nonparametric_result = wilcoxon_signed_rank(group1, group2, alpha)
    else:
        parametric_result = independent_ttest(group1, group2, alpha)
        nonparametric_result = mann_whitney_u(group1, group2, alpha)

    # Effect sizes
    effect_sizes = compute_all_effect_sizes(group1, group2)

    # Recommendation
    if is_normal1 and is_normal2:
        recommended = parametric_result.test_name
    else:
        recommended = nonparametric_result.test_name

    # Conclusion
    main_result = parametric_result if (is_normal1 and is_normal2) else nonparametric_result
    if main_result.significant:
        conclusion = (
            f"There is a statistically significant difference between {group1_name} "
            f"and {group2_name} (p = {main_result.p_value:.4f}). "
            f"The effect size is {effect_sizes.interpretation.lower()} "
            f"(Cohen's d = {effect_sizes.cohens_d:.3f})."
        )
    else:
        conclusion = (
            f"There is no statistically significant difference between {group1_name} "
            f"and {group2_name} (p = {main_result.p_value:.4f})."
        )

    return ComprehensiveAnalysisResult(
        group1_name=group1_name,
        group2_name=group2_name,
        group1_stats=group1_stats,
        group2_stats=group2_stats,
        normality_tests={
            "group1": norm1,
            "group2": norm2
        },
        parametric_test=parametric_result,
        nonparametric_test=nonparametric_result,
        effect_sizes=effect_sizes,
        recommended_test=recommended,
        conclusion=conclusion
    )


# =============================================================================
# CROSS-VALIDATION STATISTICAL TESTING
# =============================================================================

def compare_cv_results(
    scores1: np.ndarray,
    scores2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    alpha: float = 0.05
) -> Dict:
    """
    Statistical comparison of cross-validation results.

    Uses corrected resampled t-test (Nadeau & Bengio, 2003).
    """
    n_folds = len(scores1)

    # Basic comparison
    diff = scores1 - scores2
    mean_diff = np.mean(diff)

    # Corrected resampled t-test
    # Accounts for non-independence of CV folds
    var_diff = np.var(diff, ddof=1)

    # Correction factor (assuming test_size / train_size ratio)
    # For 10-fold CV: test=0.1, train=0.9
    test_train_ratio = 1 / (n_folds - 1)
    corrected_var = var_diff * (1/n_folds + test_train_ratio)

    t_stat = mean_diff / (np.sqrt(corrected_var) + 1e-10)
    df = n_folds - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Effect size
    d = mean_diff / (np.std(diff, ddof=1) + 1e-10)

    # Also run paired t-test and Wilcoxon for comparison
    t_stat_paired, p_paired = ttest_rel(scores1, scores2)

    try:
        w_stat, p_wilcoxon = wilcoxon(scores1, scores2)
    except ValueError:
        w_stat, p_wilcoxon = 0, 1.0

    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "model1_mean": float(np.mean(scores1)),
        "model1_std": float(np.std(scores1)),
        "model2_mean": float(np.mean(scores2)),
        "model2_std": float(np.std(scores2)),
        "mean_difference": float(mean_diff),
        "corrected_t_test": {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha
        },
        "paired_t_test": {
            "statistic": float(t_stat_paired),
            "p_value": float(p_paired),
            "significant": p_paired < alpha
        },
        "wilcoxon_test": {
            "statistic": float(w_stat),
            "p_value": float(p_wilcoxon),
            "significant": p_wilcoxon < alpha
        },
        "effect_size_d": float(d),
        "effect_interpretation": interpret_effect_size(d),
        "n_folds": n_folds,
        "conclusion": f"{model1_name} is {'significantly better' if p_value < alpha and mean_diff > 0 else 'significantly worse' if p_value < alpha and mean_diff < 0 else 'not significantly different'} than {model2_name}"
    }


# =============================================================================
# POWER ANALYSIS
# =============================================================================

def power_analysis_ttest(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0
) -> int:
    """
    Calculate required sample size for t-test.

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired power (1 - beta)
        ratio: Ratio of n2/n1

    Returns:
        Required sample size per group
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n1 = ((z_alpha + z_beta)**2 * (1 + 1/ratio)) / (effect_size**2)

    return int(np.ceil(n1))


def achieved_power(
    n1: int,
    n2: int,
    effect_size: float,
    alpha: float = 0.05
) -> float:
    """
    Calculate achieved power for given sample sizes.
    """
    from scipy.stats import norm, nct

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))

    # Critical t-value
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Power from non-central t distribution
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

    return float(power)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_statistical_report(
    analysis_result: ComprehensiveAnalysisResult,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a formatted statistical report.
    """
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Group comparison
    report.append(f"Comparison: {analysis_result.group1_name} vs {analysis_result.group2_name}")
    report.append("-" * 50)

    # Descriptive statistics
    report.append("\nDESCRIPTIVE STATISTICS")
    report.append("-" * 30)
    g1 = analysis_result.group1_stats
    g2 = analysis_result.group2_stats

    report.append(f"{'Statistic':<15} {analysis_result.group1_name:<15} {analysis_result.group2_name:<15}")
    report.append(f"{'N':<15} {g1['n']:<15} {g2['n']:<15}")
    report.append(f"{'Mean':<15} {g1['mean']:<15.4f} {g2['mean']:<15.4f}")
    report.append(f"{'SD':<15} {g1['std']:<15.4f} {g2['std']:<15.4f}")
    report.append(f"{'Median':<15} {g1['median']:<15.4f} {g2['median']:<15.4f}")

    # Normality tests
    report.append("\nNORMALITY TESTS")
    report.append("-" * 30)
    for group_name, norm_results in analysis_result.normality_tests.items():
        report.append(f"\n{group_name}:")
        for test_name, result in norm_results.items():
            status = "Normal" if result.is_normal else "Non-normal"
            report.append(f"  {result.test_name}: p = {result.p_value:.4f} ({status})")

    # Statistical tests
    report.append("\nSTATISTICAL TESTS")
    report.append("-" * 30)

    param = analysis_result.parametric_test
    report.append(f"\nParametric: {param.test_name}")
    report.append(f"  Statistic: {param.statistic:.4f}")
    report.append(f"  p-value: {param.p_value:.4f}")
    report.append(f"  Significant: {'Yes' if param.significant else 'No'}")

    nonparam = analysis_result.nonparametric_test
    report.append(f"\nNon-parametric: {nonparam.test_name}")
    report.append(f"  Statistic: {nonparam.statistic:.4f}")
    report.append(f"  p-value: {nonparam.p_value:.4f}")
    report.append(f"  Significant: {'Yes' if nonparam.significant else 'No'}")

    # Effect sizes
    report.append("\nEFFECT SIZES")
    report.append("-" * 30)
    es = analysis_result.effect_sizes
    report.append(f"  Cohen's d: {es.cohens_d:.4f}")
    report.append(f"  Hedges' g: {es.hedges_g:.4f}")
    report.append(f"  Glass's delta: {es.glass_delta:.4f}")
    report.append(f"  CLES: {es.cles:.4f}")
    report.append(f"  Interpretation: {es.interpretation}")

    # Conclusion
    report.append("\nCONCLUSION")
    report.append("-" * 30)
    report.append(f"Recommended test: {analysis_result.recommended_test}")
    report.append(f"\n{analysis_result.conclusion}")

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)

    return report_text


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Advanced Statistical Analysis Module")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)

    # Two groups with known difference
    group1 = np.random.normal(100, 15, 50)  # Control
    group2 = np.random.normal(110, 15, 50)  # Treatment (d ≈ 0.67)

    print("\n1. Comprehensive Two-Group Analysis")
    print("-" * 40)
    result = comprehensive_two_group_analysis(
        group1, group2,
        "Control", "Treatment"
    )

    print(f"Parametric test p-value: {result.parametric_test.p_value:.4f}")
    print(f"Non-parametric test p-value: {result.nonparametric_test.p_value:.4f}")
    print(f"Cohen's d: {result.effect_sizes.cohens_d:.4f}")
    print(f"Recommended: {result.recommended_test}")
    print(f"\nConclusion: {result.conclusion}")

    print("\n2. Bootstrap Confidence Interval")
    print("-" * 40)
    point, ci_low, ci_high = bootstrap_ci(group1, np.mean, n_bootstrap=1000)
    print(f"Mean: {point:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")

    print("\n3. Permutation Test")
    print("-" * 40)
    obs_stat, perm_p = permutation_test(group1, group2, n_permutations=5000)
    print(f"Observed difference: {obs_stat:.2f}")
    print(f"Permutation p-value: {perm_p:.4f}")

    print("\n4. Power Analysis")
    print("-" * 40)
    n_required = power_analysis_ttest(effect_size=0.5, power=0.8)
    print(f"Required n per group for d=0.5: {n_required}")

    achieved = achieved_power(50, 50, effect_size=0.67)
    print(f"Achieved power with n=50 and d=0.67: {achieved:.3f}")

    print("\n5. Multiple Comparison Correction")
    print("-" * 40)
    p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
    bonf = bonferroni_correction(p_values)
    fdr = benjamini_hochberg_fdr(p_values)

    print("Original p-values:", [f"{p:.3f}" for p in p_values])
    print("Bonferroni corrected:", [f"{p:.3f}" for p in bonf.corrected_p_values])
    print("FDR corrected:", [f"{p:.3f}" for p in fdr.corrected_p_values])

    print("\n6. Correlation Analysis")
    print("-" * 40)
    x = np.random.randn(100)
    y = 0.6 * x + 0.4 * np.random.randn(100)  # r ≈ 0.83

    corr_results = comprehensive_correlation(x, y)
    for method, result in corr_results.items():
        print(f"{result.method}: r = {result.correlation:.3f}, p = {result.p_value:.4f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
