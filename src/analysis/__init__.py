"""
Analysis Module for GenAI-RAG-EEG.

This module provides comprehensive data analysis and statistical testing
capabilities for EEG-based stress classification.

Modules:
    - signal_analysis: EEG signal processing and band power analysis
    - statistical_analysis: Advanced statistical tests and effect sizes
    - data_analysis: Comprehensive data loading and analysis pipeline

Reference: GenAI-RAG-EEG Paper v2, IEEE Sensors Journal 2024
"""

from .signal_analysis import (
    compute_psd,
    compute_band_power,
    band_power_analysis,
    alpha_suppression_analysis,
    theta_beta_ratio_analysis,
    frontal_asymmetry_analysis,
    compute_all_metrics,
    run_complete_signal_analysis,
    BandPowerResult,
    ClassificationMetrics,
    FREQUENCY_BANDS,
    CHANNEL_GROUPS
)

from .statistical_analysis import (
    # Effect sizes
    cohens_d,
    hedges_g,
    glass_delta,
    common_language_effect_size,
    eta_squared,
    omega_squared,
    compute_all_effect_sizes,

    # Normality tests
    test_normality,
    check_assumptions,

    # Parametric tests
    independent_ttest,
    paired_ttest,
    one_way_anova,

    # Non-parametric tests
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    friedman_test,
    mcnemar_test,

    # Multiple comparisons
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_fdr,

    # Correlation
    comprehensive_correlation,

    # Bootstrap and permutation
    bootstrap_ci,
    permutation_test,

    # Comprehensive analysis
    comprehensive_two_group_analysis,
    compare_cv_results,

    # Power analysis
    power_analysis_ttest,
    achieved_power,

    # Report generation
    generate_statistical_report,

    # Data classes
    StatisticalTestResult,
    NormalityTestResult,
    CorrelationResult,
    EffectSizeResult,
    MultipleComparisonResult,
    ComprehensiveAnalysisResult
)

from .data_analysis import (
    EEGDataLoader,
    QualityAssessor,
    FeatureExtractor,
    EEGAnalyzer,
    DatasetInfo,
    QualityReport,
    FeatureSet,
    AnalysisResult
)

from .visualization import (
    plot_band_power_comparison,
    plot_band_power_heatmap,
    plot_violin_comparison,
    plot_effect_size_forest,
    plot_significance_volcano,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_multi_roc,
    plot_cross_validation_results,
    plot_psd,
    plot_spectrogram,
    plot_analysis_summary
)

__all__ = [
    # Signal analysis
    'compute_psd',
    'compute_band_power',
    'band_power_analysis',
    'alpha_suppression_analysis',
    'theta_beta_ratio_analysis',
    'frontal_asymmetry_analysis',
    'compute_all_metrics',
    'run_complete_signal_analysis',
    'BandPowerResult',
    'ClassificationMetrics',
    'FREQUENCY_BANDS',
    'CHANNEL_GROUPS',

    # Statistical analysis
    'cohens_d',
    'hedges_g',
    'glass_delta',
    'common_language_effect_size',
    'eta_squared',
    'omega_squared',
    'compute_all_effect_sizes',
    'test_normality',
    'check_assumptions',
    'independent_ttest',
    'paired_ttest',
    'one_way_anova',
    'mann_whitney_u',
    'wilcoxon_signed_rank',
    'kruskal_wallis',
    'friedman_test',
    'mcnemar_test',
    'bonferroni_correction',
    'holm_bonferroni_correction',
    'benjamini_hochberg_fdr',
    'comprehensive_correlation',
    'bootstrap_ci',
    'permutation_test',
    'comprehensive_two_group_analysis',
    'compare_cv_results',
    'power_analysis_ttest',
    'achieved_power',
    'generate_statistical_report',
    'StatisticalTestResult',
    'NormalityTestResult',
    'CorrelationResult',
    'EffectSizeResult',
    'MultipleComparisonResult',
    'ComprehensiveAnalysisResult',

    # Data analysis
    'EEGDataLoader',
    'QualityAssessor',
    'FeatureExtractor',
    'EEGAnalyzer',
    'DatasetInfo',
    'QualityReport',
    'FeatureSet',
    'AnalysisResult'
]
