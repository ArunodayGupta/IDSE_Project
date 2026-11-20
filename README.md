# DataLab: TMDB Movie Worldwide Box Office Revenue Prediction

An end-to-end machine learning platform predicting worldwide theatrical box-office revenue from pre-release metadata, built on the Kaggle TMDB (The Movie Database) Box Office Prediction dataset.

The platform combines an interactive **Streamlit** dashboard, a self-contained 17-section **Jupyter Notebook**, and a production-grade **Scikit-Learn / LightGBM** machine learning pipeline designed with strict target-leakage prevention and reproducible cross-validation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Statistics & Key Numbers](#2-project-statistics--key-numbers)
3. [Jupyter Notebook Documentation](#3-jupyter-notebook-documentation)
4. [Streamlit Application Architecture](#4-streamlit-application-architecture)
5. [Complete Machine Learning Pipeline](#5-complete-machine-learning-pipeline)
6. [Feature Engineering & Leakage Prevention](#6-feature-engineering--leakage-prevention)
7. [Model Comparison & Benchmark Results](#7-model-comparison--benchmark-results)
8. [Evaluation Model vs. Production Model](#8-evaluation-model-vs-production-model)
9. [Technologies Used](#9-technologies-used)
10. [Repository Structure](#10-repository-structure)
11. [Installation & Execution Instructions](#11-installation--execution-instructions)
12. [Project Limitations & Analytical Caveats](#12-project-limitations--analytical-caveats)

---

## 1. Project Overview

### Problem Statement
Predicting the worldwide commercial success of a theatrical film is one of the highest-stakes challenges in the entertainment industry. Major motion pictures require tens or hundreds of millions of dollars in capital commitments years before opening weekend. 

**DataLab** addresses this problem by formulating box-office prediction as a **supervised regression task**: given only pre-release metadata known prior to theatrical distribution (e.g., production budget, release timing, creative talent, studio backing, genre classifications, and marketing indicators), predict the film's **worldwide theatrical box-office revenue in USD**.

### Supervised Learning Formulation
- **Input Feature Vector ($\mathbf{x}$):** 37 canonical pre-release features derived from TMDB movie metadata.
- **Target Variable ($y$):** Worldwide box-office gross receipts (`revenue`) in USD.
- **Target Transformation:** Because theatrical box office spans eight orders of magnitude (from \$1,000 to over \$1.5 billion), models are trained on log-transformed revenue:
  $$z = \ln(1 + y)$$
- **Primary Optimization Metric:** **Root Mean Squared Logarithmic Error (RMSLE)**, which penalizes proportional relative errors rather than absolute dollar deviations, preventing multi-billion-dollar blockbusters from dominating model loss:
  $$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\ln(1 + \hat{y}_i) - \ln(1 + y_i)\right)^2}$$
- **Predictions in USD:** Converted back to dollar space via the inverse exponential transformation with non-negative clipping:
  $$\hat{y} = \max\left(0, \, e^{\hat{z}} - 1\right)$$

### Dataset
The project utilizes the official Kaggle **TMDB Box Office Prediction** benchmark dataset, consisting of historical feature films with rich semi-structured JSON metadata representing production companies, countries, spoken languages, cast, and crew credits.

---

## 2. Project Statistics & Key Numbers

| Dimension | Specification |
|---|---|
| **Labeled Training Records** | **3,000 movies** with verified worldwide revenue (`train.csv`) |
| **Competition Test Records** | **4,398 unlabelled movies** for Kaggle submission (`test.csv`) |
| **Engineered Input Features** | **37 canonical pre-release features** (23 base features + 14 genre indicators) |
| **Transformed Feature Matrix** | **69 columns** after one-hot encoding categorical variables |
| **Holdout Partition Split** | **80:20 train/holdout split** (2,400 training / 600 holdout records locked with seed `42`) |
| **Cross-Validation Strategy** | **5-Fold Cross-Validation** with leak-free fold-level preprocessing |
| **Baseline Architectures** | **7 model families** evaluated on default parameters |
| **Tuned Model Families** | **6 model families** tuned over 30 iterations via `RandomizedSearchCV` |
| **Ensemble Architecture** | **Stacking Ensemble** blending out-of-fold predictions with RidgeCV meta-learner |
| **Production Champion Model** | **Tuned LightGBM Regressor** serialized to `best_movie_revenue_model.joblib` |

---

## 3. Jupyter Notebook Documentation

The project includes a self-contained, reproducible notebook located at `notebooks/exploratory_analysis.ipynb`. It executes the end-to-end data science lifecycle across **17 sequential sections**:

1. **Section 1: Project Introduction & Supervised Learning Formulation** — Defines the theatrical revenue prediction objective, log-target transformation mathematics, and RMSLE optimization formulation.
2. **Section 2: Environment Setup & Package Requirements** — Verifies boosting dependencies (LightGBM, XGBoost, CatBoost, SHAP, Scikit-Learn) and configures global seeds (`RANDOM_STATE = 42`).
3. **Section 3: Dataset Loading & Directory Resolution** — Implements anchored filesystem discovery resolving `train.csv` and `test.csv` across standard repository paths.
4. **Section 4: Dataset Profiling & Understanding** — Audits data types, missing value percentages, descriptive statistics, and quantifies zero-budget missingness (812 movies, 27.1%).
5. **Section 5: Data Cleaning & Parsing** — Implements safe JSON parsing for nested metadata columns and century-safe date parsing for 2-digit release years.
6. **Section 6: Exploratory Data Analysis (EDA)** — Renders 10 foundational visualizations (Charts 1–10) examining target skewness, log normalization, budget correlations, genre medians, historical box-office growth, release month seasonality, country distribution, and theatrical runtime sweet-spots.
7. **Section 7: Feature Engineering — 37 Pre-Release Features** — Implements the canonical feature extraction pipeline converting raw JSON metadata into 23 base features and 14 binary genre indicators.
8. **Section 8: Target Leakage Prevention** — Programmatic audit ensuring zero post-release metrics (`popularity`, `vote_count`, `vote_average`) or raw identifiers enter the feature matrices.
9. **Section 9: Train and Holdout Validation Split** — Partitions 3,000 labeled records into an 80% train split ($N = 2,400$) and an independent 20% holdout split ($N = 600$) with derived partition statistics.
10. **Section 10: Preprocessing Pipeline Construction** — Constructs scikit-learn `ColumnTransformer` with median numerical imputation, standard scaling, and one-hot encoding for categorical variables.
11. **Section 11: 5-Fold Cross-Validation & Model Comparison (7 Architectures)** — Executes 5-fold cross-validation across 7 baseline models with fold-level preprocessing to prevent data leakage.
12. **Section 12: Model Comparison Leaderboard & Cross-Validation Visualizations** — Summarizes baseline CV performance and renders Chart 11 (CV RMSLE with fold standard deviation error bars).
13. **Section 13: Kaggle Research Benchmarks & Hyperparameter Tuning for All Six ML Models** — Performs 30-iteration `RandomizedSearchCV` for Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, and CatBoost; displays Chart 12 (Baseline vs. Tuned CV RMSLE).
14. **Section 14: Model Performance Comparison & Kaggle-Inspired Stacking Ensemble** — Benchmarks 14 candidate configurations, trains an out-of-fold RidgeCV Stacking Ensemble, exports benchmark CSV artifacts, and renders Graphs A–D and Chart 13 (Holdout Residual Distribution).
15. **Section 15: Revenue Drivers, Feature Interactions & Model Explainability** — Renders Chart 14 (Franchise vs. Standalone multiple analysis), Chart 15 (Genre-Budget Heatmap), Chart 16 (SHAP Beeswarm Plot), and Chart 17 (Top 15 Mean |SHAP| Bar Chart).
16. **Section 16: Interactive Single-Movie Prediction** — Tests the champion pipeline on real-world test scenarios (mid-budget studio blockbuster vs. zero-budget indie movie).
17. **Section 17: Final Findings, Limitations & Research Notes** — Synthesizes architectural insights, model selection trade-offs, and entertainment industry business realities.

---

## 4. Streamlit Application Architecture

The web application is launched via `python -m streamlit run app.py` and presents three consolidated sections accessible from the sidebar:

### 1. Overview Dashboard
An educational and executive interface introducing the dataset, financial KPIs, and revenue dynamics:
- **KPI Summary Cards:** Total Movies (3,000), Average Worldwide Revenue (\$66.7M), Median Revenue (\$16.8M), Median Budget (\$16.0M), and Median Gross Multiple (2.29x).
- **Revenue Distribution:** Interactive toggle between raw USD receipts (extreme right-skew) and log-transformed revenue (near-normal distribution).
- **Budget vs. Revenue Scatter:** Log-log scatter plot with ordinary least squares trendline and 1.0x break-even parity reference line.
- **Revenue by Genre:** Horizontal bar chart of median box-office gross across 19 genres.
- **Budget Distribution:** Histogram of production budgets highlighting capital tiers.
- **Historical Revenue Trends:** Longitudinal tracking of box-office growth and inflation from 1980 through 2018.
- **Franchise vs. Standalone Analysis:** Evaluates the multiple advantage of collection films (3.63x median multiple) versus standalone films (1.56x).
- **Genre vs. Budget Heatmap:** Bivariate interaction heatmap analyzing median returns across budget quartiles (Q1–Q4).
- **SHAP Model Explainability:** Interactive bar chart quantifying the top feature attributions computed via `shap.TreeExplainer` on the production LightGBM booster.

### 2. Data Explorer
An interactive data hygiene and transformation laboratory:
- **Dataset Switching:** Instant toggling between the 3,000-movie training dataset (`train.csv`) and the 4,398-movie competition test dataset (`test.csv`).
- **Data Filtering & Search:** Title keyword search, release year range slider, genre multi-select, and budget/revenue boundaries.
- **Data Quality Audits:** Real-time missing-value rankings, null percentages, and duplicate record checks.
- **Descriptive Statistics:** Full numerical summaries (`count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`).
- **Interactive Cleaning Pipeline:** Checkbox-driven simulation of 7 data hygiene operations (dropping zero-revenue records, imputing zero budgets, handling missing runtimes, binary collection flags, studio categorization, and log transformation).
- **Before-and-After Statistical Comparison:** Direct comparison metrics verifying skewness compression, row deltas, and null count reductions.
- *(Note: Data Explorer is strictly an exploratory tool; it does not provide CSV export functionality).*

### 3. ML Studio
The model benchmarking, diagnostics, and real-time prediction workspace:
- **Model Comparison Table:** Complete 14-configuration leaderboard reporting 5-Fold CV RMSLE (Mean $\pm$ Std), Holdout RMSLE, Holdout MAE, Holdout $R^2$, and Training Duration.
- **Benchmark Visualizations Grid (2x2 Balanced Layout):**
  - **Graph A:** 5-Fold CV Mean RMSLE with standard deviation error bars vs. Holdout RMSLE (Lower is Better).
  - **Graph B:** Holdout Mean Absolute Error in USD across candidate models (Lower is Better).
  - **Graph C:** Holdout Explained Variance $R^2$ across candidate models (Higher is Better).
  - **Graph D:** Baseline vs. Tuned 5-Fold CV RMSLE across the six machine learning model families.
- **Benchmark Findings Callout:** Dynamically calculated summary highlighting key metric winners and hyperparameter tuning gains.
- **Holdout Evaluation Diagnostics:**
  - Prediction Error Parity Plot ($y = x$ reference line).
  - Residuals vs. Fitted Values scatter plot.
  - Trimmed Holdout Residual Distribution (1st–99th percentile histogram with zero-error line).
  - Feature Importance Bar Chart extracted from tree booster weights.
- **Interactive Single-Movie Prediction Form:** Live revenue estimator allowing users to specify budget, runtime, release date, genres, spoken language, production country, studio backing, franchise collection, and talent sizes, generating instant dollar estimates and gross budget multiples.
- *(Note: ML Studio is an evaluation and inference interface; it does not provide in-session model retraining controls).*

---

## 5. Complete Machine Learning Pipeline

The project implements an end-to-end 14-stage machine learning workflow:

```
[Raw TMDB Data] 
       │
       ▼
1. Dataset Loading ───────────► 2. Cleaning & JSON Parsing ────► 3. Feature Engineering (37 Features)
                                                                             │
                                                                             ▼
6. Preprocessing (Impute/Scale) ◄── 5. Locked 80:20 Split ◄───── 4. Target Leakage Prevention
       │
       ▼
7. Log Transformation ln(1+y) ──► 8. Baseline Benchmarking (7) ─► 9. 5-Fold Cross-Validation
                                                                             │
                                                                             ▼
12. Holdout Evaluation (20%) ◄── 11. Stacking Ensemble ◄──────── 10. RandomizedSearchCV (30 Iters)
       │
       ▼
13. Production Artifact Export ──► 14. Interactive Single-Movie Prediction
```

1. **Dataset Loading:** Anchored discovery resolves `train.csv` and `test.csv` across environments.
2. **Cleaning & JSON Parsing:** Robust parsing of malformed stringified dictionary arrays (`genres`, `production_companies`, `cast`, `crew`) and century date correction.
3. **Feature Engineering:** Extracts 37 pre-release signals capturing financial, temporal, genre, studio, talent, and marketing attributes.
4. **Target Leakage Prevention:** Programmatically drops post-release signals (`popularity`, `vote_count`, `vote_average`) and uninformative IDs.
5. **Train/Holdout Split:** Locks an independent 20% holdout split ($N = 600$) with seed `42`; all training and tuning statistics are calculated strictly on the 80% train split ($N = 2,400$).
6. **Numerical & Categorical Preprocessing:** Scikit-Learn `ColumnTransformer` applies median imputation and standard scaling to numeric predictors, and constant imputation with one-hot encoding (`handle_unknown="ignore"`) to categoricals.
7. **Logarithmic Target Transformation:** Fits models on $z = \ln(1 + y)$ to stabilize exponential variance and align with the competition RMSLE loss metric.
8. **Baseline Model Comparison:** Trains 7 diverse algorithm families on default parameters to establish performance baselines.
9. **Five-Fold Cross-Validation:** Executes 5-fold CV fitting preprocessors strictly inside each fold to ensure leak-free evaluation.
10. **Randomized Hyperparameter Tuning:** Explores 30 parameter combinations across 6 model families using negative RMSLE scoring.
11. **OOF-Based Stacking Ensemble:** Generates out-of-fold cross-validation predictions from top tuned estimators and trains a `RidgeCV` meta-regressor.
12. **Holdout Evaluation:** Measures unbiased out-of-sample generalization across all 14 candidate configurations on the locked 600-movie holdout partition.
13. **Production Model Saving:** Fits the champion LightGBM architecture on 100% of labeled data with optimal hyperparameters, serializing the full pipeline to `best_movie_revenue_model.joblib`.
14. **Single-Movie Prediction:** Live inference function transforms arbitrary user inputs and computes predicted gross revenue and budget multiples.

---

## 6. Feature Engineering & Leakage Prevention

The feature engineering module (`src/feature_engineering.py`) extracts **37 canonical pre-release features** grouped into 10 domain categories:

| Feature Category | Features Included | Description & Business Rationale |
|---|---|---|
| **1. Financial Features** | `budget`, `log_budget`, `budget_is_zero` | Raw production budget in USD, log1p-transformed budget, and binary missing-budget indicator flag. |
| **2. Release Timing** | `release_year`, `release_month`, `release_quarter`, `release_dayofweek`, `release_is_weekend` | Captures long-term inflation/growth, summer/holiday seasonality, and weekend opening dynamics. |
| **3. Genre Classification** | `primary_genre`, `genre_count`, and 14 binary genre flags | Lead genre name, total genres credited, and one-hot indicators (`genre_action`, `genre_comedy`, `genre_drama`, etc.). |
| **4. Studio Backing** | `company_count`, `is_major_studio` | Total producing studios and binary flag for major Hollywood distribution (Warner Bros, Universal, Paramount, Disney, Sony, etc.). |
| **5. Geographic Scope** | `country_count`, `is_us_country` | Total production countries and binary indicator for United States domestic production. |
| **6. Language Metadata** | `original_language`, `language_count`, `is_english` | Primary language code, total spoken languages, and binary flag for English-language dominance. |
| **7. Cast & Crew Scale** | `cast_size`, `crew_size` | Total credited on-screen actors and credited production crew members, serving as a proxy for production scale. |
| **8. Franchise Status** | `has_collection` | Binary flag indicating whether the film belongs to an established movie collection, sequel, or universe. |
| **9. Marketing Indicators** | `has_homepage`, `has_tagline` | Binary flags indicating dedicated digital marketing presence and commercial tagline promotional assets. |
| **10. Theatrical Runtime** | `runtime` | Theatrical running time in minutes (median-imputed). |

### Preprocessing Dimension Expansion
During preprocessing, `original_language` and `primary_genre` are one-hot encoded by the `ColumnTransformer`. This expands the feature matrix from the 37 raw input features to approximately **69 numerical columns** supplied to the estimators.

### Target Leakage Prevention Guarantee
To ensure models evaluate real-world pre-release predictive power, all post-release signals present in the raw TMDB dataset are strictly purged prior to modeling:
- `popularity` (incorporates post-release viewings and search traffic).
- `vote_count` and `vote_average` (reflect audience ratings submitted after theatrical release).
- Raw database IDs (`id`, `imdb_id`, `poster_path`, `overview`, `status`).

---

## 7. Model Comparison & Benchmark Results

All models were evaluated using 5-fold cross-validation on the 2,400-movie training partition and tested on the untouched 600-movie holdout partition. 

The metrics below match the persistent evaluation artifact (`data/model_comparison_results.csv`) exactly:

| Model Architecture | Model Key | Is Tuned | 5-Fold CV RMSLE | Holdout RMSLE | Holdout MAE ($ USD) | Holdout R² | Training Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stacking Ensemble (Top 3 OOF Blend)** | `stacking_ensemble` | True | **2.1486 ± 0.0962** | 1.9982 | \$37,126,558 | **0.6708** | 0.02s |
| **LightGBM (Tuned)** | `lightgbm` | True | **2.1576 ± 0.0828** | 2.0192 | **\$37,055,916** | **0.6662** | 4.24s |
| **Random Forest (Tuned)** | `random_forest` | True | 2.1636 ± 0.0925 | 2.0227 | \$37,594,319 | 0.6623 | 1.67s |
| **HistGradientBoosting (Tuned)** | `hist_gradient_boosting` | True | 2.1740 ± 0.1165 | 2.0460 | \$39,301,261 | 0.6105 | 4.60s |
| **XGBoost (Tuned)** | `xgboost` | True | 2.1749 ± 0.1108 | 2.0099 | \$38,219,508 | 0.6618 | 0.60s |
| **Extra Trees (Tuned)** | `extra_trees` | True | 2.1753 ± 0.1002 | 2.0084 | \$38,369,129 | 0.6585 | 2.18s |
| **CatBoost (Tuned)** | `catboost` | True | 2.1761 ± 0.1124 | **1.9981** | \$40,217,785 | 0.6109 | 2.53s |
| **Extra Trees (Baseline)** | `extra_trees` | False | 2.2654 ± 0.1120 | **1.9926** | \$37,917,548 | 0.6669 | 0.53s |
| **Random Forest (Baseline)** | `random_forest` | False | 2.2081 ± 0.0909 | 2.0241 | \$37,769,439 | 0.6632 | 0.52s |
| **XGBoost (Baseline)** | `xgboost` | False | 2.2006 ± 0.1180 | 2.0369 | \$39,132,848 | 0.6194 | 0.26s |
| **LightGBM (Baseline)** | `lightgbm` | False | 2.2067 ± 0.1096 | 2.0845 | \$38,338,242 | 0.6483 | 0.39s |
| **HistGradientBoosting (Baseline)** | `hist_gradient_boosting` | False | 2.1992 ± 0.1146 | 2.0592 | \$38,997,927 | 0.6501 | 1.41s |
| **CatBoost (Baseline)** | `catboost` | False | 2.2326 ± 0.1316 | 2.0719 | \$43,140,491 | 0.4962 | 0.45s |
| **Dummy Baseline (Median)** | `dummy` | False | 3.1420 ± 0.2007 | 3.0203 | \$66,076,091 | -0.1811 | 0.05s |

### Architectural Trade-Offs & Model Selection
- **Cross-Validation Winner:** **Stacking Ensemble** achieves the lowest 5-fold CV RMSLE (**$2.1486$**) and highest holdout explained variance (**$R^2 = 0.6708$**), confirming the power of blending diverse tree structures.
- **Holdout RMSLE Winner:** **Extra Trees Baseline** ($1.9926$) and **CatBoost Tuned** ($1.9981$) exhibit superior tail robustness on extreme holdout observations.
- **Dollar Precision Winner (MAE):** **Tuned LightGBM** achieves the lowest holdout Mean Absolute Error (**\$37.06M**), outperforming Stacking (\$37.13M), Random Forest (\$37.59M), and Extra Trees (\$37.92M).
- **Why LightGBM was Selected for Production:**
  1. *Dollar Accuracy:* Secures the lowest absolute error in original currency units.
  2. *Single-Model Simplicity:* Avoids the latency and multi-framework runtime overhead of running a 6-model stacking pipeline.
  3. *Interpretability:* Natively interfaces with `shap.TreeExplainer` for exact Shapley value decomposition in the dashboard.

---

## 8. Evaluation Model vs. Production Model

To maintain scientific integrity and prevent data snooping, the project strictly distinguishes the evaluation models from the deployed production artifact:

- **The Evaluation Models (Benchmark Leaderboard):**
  - Trained strictly on the **80% training split (2,400 movies)**.
  - Evaluated on the **locked 20% holdout split (600 movies)**.
  - Used to generate `data/model_comparison_results.csv`, `data/model_holdout_predictions.csv`, and all diagnostic error curves in ML Studio.
- **The Production Model (`best_movie_revenue_model.joblib`):**
  - Built with the identical winning LightGBM architecture and tuned hyperparameters.
  - **Retrained on 100% of available labeled data (3,000 movies)** solely to maximize predictive quality and historical pattern coverage for future interactive predictions.
  - **Integrity Rule:** The production model is *never* evaluated on the training set to present artificial holdout scores. The dashboard displays holdout metrics computed exclusively from the genuine untouched holdout partition.

---

## 9. Technologies Used

- **Python 3.12:** Primary programming language.
- **Pandas & NumPy:** Tabular data processing, linear algebra, vectorization, and data profiling.
- **Scikit-Learn:** Machine learning pipelines, `ColumnTransformer`, imputation, scaling, one-hot encoding, K-fold cross validation, and `RidgeCV`.
- **LightGBM:** Fast, distributed, high-performance gradient boosting framework using histogram-based tree learning (champion production model).
- **XGBoost:** Extreme Gradient Boosting regressor with histogram tree methods.
- **CatBoost:** Fast gradient boosting handling high-cardinality categoricals with Bayesian bootstrapping.
- **Plotly & Plotly Express:** Interactive, publication-grade data visualizations with responsive tooltips and custom themes.
- **SHAP (SHapley Additive exPlanations):** Cooperative game-theoretic interpretability calculating exact feature attribution via `TreeExplainer`.
- **Joblib:** Pipeline serialization, multi-directory artifact resolution, and parallel execution.
- **Streamlit:** Reactive web framework providing interactive sliders, tabs, dataframes, and KPI scorecards.
- **Matplotlib:** Backend figure generation for complex SHAP beeswarm visualizations.
- **Jupyter Notebook:** Interactive research environment for exploratory data analysis and model benchmarking.
- **Git & GitHub:** Version control, automated tracking, and source code management.

---

## 10. Repository Structure

```text
IDSE_Project/
├── app.py                             # Streamlit application entry point and tab navigation
├── requirements.txt                   # Production environment dependencies
├── README.md                          # Comprehensive project documentation
├── best_movie_revenue_model.joblib    # Serialized 37-feature production LightGBM pipeline
├── submission.csv                     # Generated competition test predictions (4,398 movies)
├── data/
│   ├── train.csv                      # TMDB training dataset (3,000 movies with revenue)
│   ├── test.csv                       # TMDB competition test dataset (4,398 movies)
│   ├── sample_submission.csv          # Kaggle submission format template
│   ├── model_comparison_results.csv   # Locked benchmark metrics for 14 model configurations
│   └── model_holdout_predictions.csv  # Holdout actuals and predictions (600 records)
├── notebooks/
│   └── exploratory_analysis.ipynb     # 17-section self-contained EDA and ML experiment notebook
├── src/
│   ├── __init__.py                    # Module export definitions
│   ├── data_loader.py                 # Multi-path dataset discovery and parsing
│   ├── feature_engineering.py         # 37 pre-release feature extraction and leakage prevention
│   ├── model_trainer.py               # Model training, evaluation metrics, and prediction logic
│   ├── visualizations.py              # Plotly chart generators with consistent UI styling
│   ├── ui_theme.py                    # CSS styles, layout tokens, and UI card components
│   ├── overview_tab.py                # Overview dashboard tab implementation
│   ├── data_explorer_tab.py           # Data Explorer dashboard tab implementation
│   └── ml_studio_tab.py               # ML Studio dashboard tab implementation
└── tests/
    ├── __init__.py
    ├── test_data_pipeline.py          # Unit tests for loading, JSON parsing, and date parsing
    ├── test_feature_engineering.py    # Unit tests for 37 features and zero leakage
    └── test_models.py                 # Unit tests for models, metrics, and prediction pipelines
```

---

## 11. Installation & Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ArunodayGupta/IDSE_Project.git
cd IDSE_Project
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

### 4. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### 5. Run the Automated Test Suite
```bash
python -m pytest tests -q
```

### Dataset File Discovery
The data loading utility (`src/data_loader.py`) automatically discovers dataset files by checking:
1. Current working directory (`train.csv`, `test.csv`)
2. `data/` subdirectory (`data/train.csv`, `data/test.csv`)
3. `tmdb-box-office-prediction/` subdirectory
4. Parent directory fallbacks (`../data/`, `../`)

---

## 12. Project Limitations & Analytical Caveats

While the predictive models demonstrate competitive cross-validation and holdout metrics, several real-world entertainment domain limitations apply:

1. **Limited Sample Size:** The labeled training dataset comprises 3,000 films. In high-dimensional feature spaces with rare genres (e.g., Foreign, Documentary), niche languages, and smaller production companies, sample support is limited.
2. **Missing & Sentinel Metadata:** 27.1% of films report a budget of \$0, which represents unrecorded expenditure rather than free production. While median imputation and binary missingness indicators mitigate bias, unrecorded budgets introduce residual variance.
3. **Unobserved Marketing & Distribution:** Theatrical revenue is heavily driven by **Prints and Advertising (P&A) marketing budgets** (often \$50M–\$150M for major releases) and **opening screen counts**. These variables are proprietary studio data absent from public TMDB records.
4. **Competitive & Macro Dynamics:** Opening weekend box office is strongly influenced by competing theatrical releases, major sporting events, critical review aggregations (Rotten Tomatoes), and unexpected cultural trends not captured in pre-release metadata.
5. **Historical Inflation & Secular Shifts:** The dataset spans release years from the 1920s to the 2010s. While `release_year` models secular expansion, historical ticket price inflation, international theater expansion, and modern streaming cannibalization introduce temporal non-stationarity.
6. **Extrapolation Limits:** Predictions for unprecedented mega-budget films ($>$ \$250M) or films with extreme cast sizes rely on model interpolation and exhibit wider prediction intervals.
7. **Gross Box Office vs. Studio Profit:** Worldwide gross revenue is **not net profit**. Theatrical exhibitors retain approximately 45%–55% of ticket sales, and negative production budget excludes marketing overhead, distribution fees, financing costs, and backend talent profit participation.
