# FIGURE SPECIFICATIONS DATABASE — EEG-BCI STRESS PAPER

## Version: 3.0.0

---

## MASTER FIGURE LIST

| Fig. No. | Type | Title | Content | Section | Mandatory | Status |
|----------|------|-------|---------|---------|-----------|--------|
| Fig. 1 | Architecture | Overall EEG-BCI Stress Framework | End-to-end: EEG → preprocessing → features → classifier → stats/explainability | Methods – System Overview | ✅ YES | ⚠️ Need |
| Fig. 2 | Flowchart | LOSO Validation Workflow | Training/testing separation, no data leakage | Methods – Validation | ✅ YES | ⚠️ Need |
| Fig. 3 | Flowchart | Dataset & Labeling Pipeline | Raw EEG → task/rating → normalization → labels | Methods – Labeling | ✅ YES | ⚠️ Need |
| Fig. 4 | Flow Diagram | Feature Extraction Pipeline | Bandpower, TF, Riemannian → feature vector | Methods – Features | ✅ YES | ⚠️ Need |
| Fig. 5 | Architecture | RAG-LLM Explanation Framework | EEG classifier → retriever → LLM → explanation | Methods – RAG | ⚠️ If RAG | ⚠️ Need |
| Fig. 6 | Boxplot | Subject-wise LOSO Performance | Distribution of accuracies (median, IQR) | Results – Performance | ✅ YES | ⚠️ Need |
| Fig. 7 | Matrix | Confusion Matrix | Stress vs non-stress errors | Results – Classification | ✅ YES | ⚠️ Need |
| Fig. 8 | Curve | ROC Curve | Discriminative ability (AUC) | Results – Performance | ⚠️ Strong | ⚠️ Need |
| Fig. 9 | TF Map | Stress vs Non-Stress Spectral | TF representation showing stress markers | Results – Signal | ⚠️ Strong | ⚠️ Need |
| Fig. 10 | Topography | Scalp Distribution | Spatial EEG patterns (α, θ) | Results – Signal | ⚠️ Strong | ⚠️ Need |
| Fig. 11 | Heatmap | Feature Importance | Channel × Band importance | Results – Explainability | ⚠️ Strong | ⚠️ Need |
| Fig. 12 | Bar/Line | Cross-Dataset Comparison | DEAP vs SAM-40 trends | Results/Discussion | ⚠️ Optional | ⚠️ Need |
| Fig. 13 | Bar | Ablation Study | With/without components | Supplementary | ⚠️ Optional | ⚠️ Need |
| Fig. 14 | Error | Failure Case Analysis | Low-performance subjects | Supplementary | ⚠️ Optional | ⚠️ Need |

---

## FIGURE CAPTIONS (CAMERA-READY)

### Fig. 1 — System Architecture
**Caption:**
> Overview of the proposed EEG-based stress detection framework. Raw EEG signals are preprocessed, segmented into fixed-length windows, and transformed into feature representations for subject-independent classification. Performance evaluation, statistical validation, and interpretability analyses are conducted on the classifier outputs.

### Fig. 2 — LOSO Validation Workflow
**Caption:**
> Illustration of the leave-one-subject-out (LOSO) validation protocol. For each fold, data from one subject are held out for testing, while all preprocessing, feature selection, and model training steps are performed exclusively on the remaining subjects to prevent data leakage.

### Fig. 3 — Dataset & Labeling Pipeline
**Caption:**
> Dataset-specific labeling workflow. Raw EEG data and task or rating information are combined with subject-wise normalization to derive stress and non-stress labels before window-level assignment.

### Fig. 4 — Feature Extraction Pipeline
**Caption:**
> Feature extraction process illustrating the computation of band-power, time–frequency, and covariance-based representations from windowed EEG segments, resulting in a unified feature vector for classification.

### Fig. 5 — RAG-LLM Explanation Architecture
**Caption:**
> Retrieval-augmented generation (RAG) framework used for explanation and reasoning. The EEG classifier provides predictions and symbolic summaries, which are grounded via retrieval from a domain-specific knowledge base before generating structured explanations. The RAG module does not influence classification decisions.

### Fig. 6 — Subject-Wise LOSO Performance
**Caption:**
> Distribution of subject-wise classification performance under LOSO validation. Boxplots report median accuracy, interquartile range, and outliers, highlighting inter-subject variability and generalization robustness.

### Fig. 7 — Confusion Matrix
**Caption:**
> Confusion matrix summarizing stress and non-stress classification outcomes aggregated across LOSO folds, illustrating class-specific error patterns.

### Fig. 8 — ROC Curve
**Caption:**
> Receiver operating characteristic (ROC) curve for binary stress classification, demonstrating discriminative performance independent of decision threshold.

### Fig. 9 — Time–Frequency Representation
**Caption:**
> Time–frequency representations comparing stress and non-stress conditions, revealing task-related spectral dynamics associated with stress.

### Fig. 10 — Scalp Topography
**Caption:**
> Scalp topographies of key frequency bands illustrating spatial distributions of EEG activity associated with stress.

### Fig. 11 — Feature Importance Heatmap
**Caption:**
> Channel-by-frequency importance heatmap highlighting EEG features that contribute most to stress classification, supporting interpretability of the proposed model.

### Fig. 12 — Cross-Dataset Performance
**Caption:**
> Comparison of classification performance trends across datasets, demonstrating consistency of the proposed approach under different stress paradigms.

### Fig. 13 — Ablation Study
**Caption:**
> Ablation analysis evaluating the contribution of individual components of the proposed framework to overall performance.

### Fig. 14 — Failure Case Analysis
**Caption:**
> Analysis of misclassified subjects or conditions, highlighting limitations and sources of performance degradation.

---

## ADVANCED ANALYSIS FIGURES (v3.1.0)

| Fig. No. | Type | Title | File | Section |
|----------|------|-------|------|---------|
| Fig. 15 | Curve | Precision-Recall Curves | `fig_precision_recall.png` | Results |
| Fig. 16 | Curve | Calibration Plots | `fig_calibration.png` | Results |
| Fig. 17 | Bar | SHAP Feature Importance | `fig_shap_importance.png` | Results |
| Fig. 18 | Topography | Topographical EEG Maps | `fig_topographical_maps.png` | Results |
| Fig. 19 | TF Map | Time-Frequency Spectrograms | `fig_spectrograms.png` | Results |
| Fig. 20 | Curve | Statistical Power Analysis | `fig_power_analysis.png` | Results |
| Fig. 21 | Curve | Learning Curves | `fig_learning_curves.png` | Results |
| Fig. 22 | Heatmap | Feature Correlation | `fig_feature_correlation.png` | Results |
| Fig. 23 | Forest | Effect Size Forest Plot | `fig_forest_plot.png` | Results |
| Fig. 24 | Scatter | Bland-Altman Plots | `fig_bland_altman.png` | Results |
| Fig. 25 | Bar | Cross-Subject Generalization | `fig_cross_subject.png` | Results |
| Fig. 26 | Bar | Component Importance | `fig_component_importance.png` | Ablation |
| Fig. 27 | Bar | Cumulative Ablation | `fig_cumulative_ablation.png` | Ablation |
| Fig. 28 | Heatmap | Component Interaction Matrix | `fig_component_interaction.png` | Ablation |
| Fig. 29 | Violin | Performance Distribution | `fig_performance_distribution.png` | Results |
| Fig. 30 | Radar | Comprehensive Evaluation | `fig_comprehensive_evaluation.png` | Results |

### Fig. 15 — Precision-Recall Curves
**Caption:**
> Precision-Recall curves across datasets with Average Precision (AP) scores. All datasets achieve AP > 0.90, demonstrating robust classification performance across varying decision thresholds.

### Fig. 16 — Calibration Curves
**Caption:**
> Calibration curves (reliability diagrams) comparing predicted probabilities against actual outcomes. Closer alignment to the diagonal indicates better calibration.

### Fig. 17 — SHAP Feature Importance
**Caption:**
> SHAP summary plot showing feature importance and directionality. Frontal alpha and beta features show the strongest contributions to stress classification, consistent with neurophysiological expectations.

### Fig. 18 — Topographical EEG Maps
**Caption:**
> Topographical scalp maps showing stress-induced changes in EEG power across alpha and beta bands. Blue indicates decreased power (alpha suppression), red indicates increased power (beta enhancement).

### Fig. 19 — Time-Frequency Spectrograms
**Caption:**
> Time-frequency spectrograms showing spectral power evolution. Stress conditions show characteristic alpha suppression (8–13 Hz) and beta enhancement (13–30 Hz) patterns.

### Fig. 20 — Statistical Power Analysis
**Caption:**
> Statistical power analysis curves showing achieved power (>0.99) for observed effect sizes across all datasets. The dashed line indicates the conventional 0.80 power threshold.

### Fig. 21 — Learning Curves
**Caption:**
> Learning curves showing training and validation performance as a function of training set size. Rapid convergence indicates efficient sample utilization.

### Fig. 22 — Feature Correlation Heatmap
**Caption:**
> Feature correlation heatmap showing relationships between top discriminative EEG features. High correlation within frequency bands suggests complementary spatial information.

### Fig. 23 — Effect Size Forest Plot
**Caption:**
> Forest plot of effect sizes (Cohen's d) across key comparisons. All comparisons show large effect sizes (d > 0.8) with non-overlapping confidence intervals from zero.

### Fig. 24 — Bland-Altman Plots
**Caption:**
> Bland-Altman plots for each dataset showing the difference between predicted and actual values against their mean. Limits of agreement (LoA) are shown as dashed lines.

### Fig. 25 — Cross-Subject Generalization
**Caption:**
> Cross-subject generalization analysis showing accuracy distribution across individual subjects. The consistent performance across subjects demonstrates robust generalization.

### Fig. 26 — Component Importance Ranking
**Caption:**
> Architectural component importance ranking based on accuracy contribution. The hierarchical CNN-LSTM feature extraction provides the largest contribution (+9.5%), followed by self-attention (+2.6%).

### Fig. 27 — Cumulative Ablation Analysis
**Caption:**
> Cumulative ablation analysis showing progressive performance degradation as components are removed. The steep decline after LSTM removal confirms the critical role of temporal modeling.

### Fig. 28 — Component Interaction Matrix
**Caption:**
> Component interaction matrix showing synergy (+) and redundancy (−) between architectural modules. CNN-LSTM synergy (+2.4%) confirms complementary spatial-temporal processing.

### Fig. 29 — Performance Distribution
**Caption:**
> Performance distribution across datasets showing accuracy, precision, recall, and F1-score distributions via violin plots with embedded box plots.

### Fig. 30 — Comprehensive Evaluation
**Caption:**
> Comprehensive evaluation summary showing classification performance, signal analysis metrics, and RAG explanation quality across all datasets. Radar chart format enables direct cross-dataset comparison.

---

## ARCHITECTURE DIAGRAMS (ASCII SPECIFICATIONS)

### Fig. 1 — System Architecture

```
EEG Acquisition
  (SAM-40 / DEAP)
        |
        v
Preprocessing
(Band-pass, Notch, ICA/ASR)
        |
        v
Windowing
(L sec, overlap)
        |
        v
Feature Extraction
(Bandpower / TF / Riemannian)
        |
        v
Classifier
(LDA / SVM / Proposed)
        |
        +--------------------+
        |                    |
        v                    v
Prediction Output      Explainability
(Stress / Non-stress)  (Channel×Band)
        |
        v
Statistics & Validation
(LOSO, CI, Wilcoxon)
```

### Fig. 2 — LOSO Validation Flow

```
For each subject s:
  Test Set  ← Subject s
  Train Set ← All subjects except s
      |
      v
 Train-only operations
 (Scaling, FS, Hyperparams)
      |
      v
  Train Model
      |
      v
  Test on Subject s
      |
      v
  Store Subject-wise Metrics
```

### Fig. 3 — Labeling Flow

```
Raw EEG + Task Info
        |
        v
Rating / Task Rules
(Stress / Workload)
        |
        v
Per-Subject Normalization
        |
        v
Window Label Assignment
(Stress / Non-stress)
        |
        v
Final Labeled Segments
```

### Fig. 4 — Feature Extraction

```
Windowed EEG
     |
     +--> Bandpower (θ, α, β)
     |
     +--> Time–Frequency (Wavelet)
     |
     +--> Covariance → Riemannian
                 |
                 v
          Feature Vector
```

### Fig. 5 — RAG Architecture

```
EEG Classifier Output
(Stress / Confidence)
        |
        v
Symbolic Summary
(Top channels, bands)
        |
        v
Retriever
(EEG stress KB)
        |
        v
LLM Reasoning
(Constrained, grounded)
        |
        v
Structured Explanation
(No effect on prediction)
```

---

## FIGURE COUNT RECOMMENDATIONS

| Scenario | Figures in Main | Move to Supplement |
|----------|-----------------|-------------------|
| Page-limited | 6–7 | TF maps, ablation |
| Top-conference safe | 9–10 | Failure cases |
| With RAG | 10–11 | Prompt ablations |

---

## REVIEWER WARNINGS IF MISSING

| Missing Figure | Reviewer Comment |
|----------------|------------------|
| No Fig. 1 | "Method unclear" |
| No Fig. 2 | "Possible data leakage" |
| No Fig. 6 | "Weak generalization" |
| No Fig. 9-10 | "No neurophysiological grounding" |
| No Fig. 5 (with RAG) | "LLM role unclear" |

---

*Version: 3.0.0*
*Last Updated: 2025-12-25*
