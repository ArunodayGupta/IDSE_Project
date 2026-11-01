# DataLab: Movie Box Office Worldwide Revenue Prediction Platform

An end-to-end data science platform predicting worldwide theatrical box office revenue from pre-release metadata, built on the Kaggle TMDB Box Office Prediction dataset.

The platform guides practitioners through the complete data science lifecycle:
- Dataset exploration and data hygiene auditing
- Pre-release feature engineering with strict target leakage prevention
- Longitudinal box office and production budget trends
- Machine learning model training and multi-algorithm comparison (RMSLE, MAE, R2)
- Interactive movie revenue prediction for upcoming films
- Kaggle test-set submission generation preserving competition identifiers

Built with pure Python, **Streamlit**, **Plotly**, and **Scikit-Learn**.

---

## Dashboard Architecture

The dashboard is organized into three consolidated sections accessible from the sidebar:

1. **Overview**: Executive box office trends, KPI scorecards (Total Movies, Average Worldwide Revenue, Median Worldwide Revenue, Median Budget, Gross Revenue / Budget Multiple), and interactive Plotly visualizations (Revenue Distribution with logarithmic scaling, Production Budget vs. Revenue with 1.0x parity line, Median Worldwide Revenue by Genre, Longitudinal Box Office Trends, and Production Budget Distribution).
2. **Data Explorer**: Dataset loading (default training dataset, custom CSV upload, or competition test dataset), interactive preview with keyword search, year range slider, and language filtering, data quality audit (missing values and descriptive statistics), and reproducible data cleaning and transformation tools with CSV export.
3. **ML Studio**: Consolidated predictive modeling workspace featuring target leakage prevention, algorithm training and comparison (Dummy Regressor baseline, Random Forest Regressor, Extra Trees Regressor, and HistGradientBoosting Regressor), validation metrics (RMSLE as primary metric, MAE in USD, R2), actual vs. predicted parity scatter plots, residual distribution, feature importance analysis, real-time interactive revenue prediction form with out-of-range warnings, and Kaggle test-set submission generator (`submission.csv`).

---

## Directory Structure

```text
IDSE_Project/
├── app.py                     # Streamlit application entry point and sidebar dispatcher
├── requirements.txt           # Environment dependencies
├── README.md                  # Project documentation
├── data/                      # Dataset repository (train.csv, test.csv, sample_submission.csv)
├── src/
│   ├── __init__.py            # Package exports
│   ├── data_loader.py         # Dataset discovery, safe JSON parsing, and century date parsing
│   ├── feature_engineering.py # Pre-release feature extraction, leakage prevention, and input builder
│   ├── model_trainer.py       # Model training on log1p(revenue), evaluation, prediction, and test submission
│   ├── visualizations.py      # Plotly figures styled with corporate dark theme
│   ├── ui_theme.py            # Central design system tokens, typography, and styles
│   ├── overview_tab.py        # Section A: Overview tab
│   ├── data_explorer_tab.py   # Section B: Data Explorer tab
│   └── ml_studio_tab.py       # Section C: ML Studio tab
└── tests/
    ├── __init__.py
    ├── test_data_pipeline.py  # Unit tests for data loading, JSON parsing, and date parsing
    ├── test_feature_engineering.py # Unit tests for feature extraction and leakage prevention
    └── test_models.py         # Unit tests for models, evaluation metrics, and test submissions
```

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

### Running Test Suite

```bash
python -m unittest discover tests
```

---

## Machine Learning Formulation

- **Problem Type**: Supervised Regression
- **Target Variable**: `revenue` (Worldwide Box Office Revenue in USD)
- **Target Transformation**: Models fit on `log1p(revenue)` to stabilize exponential variance across skewed theatrical returns; predictions converted with `expm1` and clipped to ensure non-negative revenue estimates.
- **Evaluation Metrics**:
  - **RMSLE** (Root Mean Squared Logarithmic Error) - Primary Kaggle metric.
  - **MAE** (Mean Absolute Error in USD).
  - **R2** (Coefficient of Determination).
- **Leakage Prevention**: Strictly excludes post-release attributes (`revenue`, `popularity`, `vote_average`, `vote_count`) and identifiers (`id`, `imdb_id`, `poster_path`, `homepage`).
