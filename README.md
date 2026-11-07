# DataLab: Movie Box Office Worldwide Revenue Prediction Platform

An end-to-end data science platform predicting worldwide theatrical box office revenue from pre-release metadata, built on the Kaggle TMDB Box Office Prediction dataset.

The platform guides practitioners through the complete data science lifecycle:
- Dataset exploration and data hygiene auditing
- Pre-release feature engineering with strict target leakage prevention
- Longitudinal box office and production budget trends
- Machine learning model training and multi-algorithm comparison (RMSLE, MAE, R2)
- Interactive movie revenue prediction for upcoming films with outlier warnings
- Kaggle test-set submission generation preserving competition identifiers

Built with pure Python, **Streamlit**, **Plotly**, and **Scikit-Learn**.

---

## Dashboard Architecture

The dashboard is organized into three consolidated sections accessible from the sidebar:

1. **Overview**: Executive box office trends, KPI scorecards (Total Movies, Average Worldwide Revenue, Median Worldwide Revenue, Median Movie Budget, Median Gross Multiple), and interactive Plotly visualizations (Revenue Distribution with logarithmic scaling, Production Budget vs. Revenue with 1.0x parity reference line, Median Worldwide Revenue by Genre, Longitudinal Box Office Trends, and Production Budget Distribution).
2. **Data Explorer**: Dataset loading (default training dataset, custom CSV upload, or competition test dataset), interactive preview with keyword search, dynamic release year range slider, and language filtering, data quality audit (missing values and descriptive statistics), and reproducible data cleaning and transformation tools with CSV export.
3. **ML Studio**: Consolidated predictive modeling workspace featuring target leakage prevention, leak-free train/validation splits, algorithm training and comparison (Dummy Regressor baseline, Random Forest Regressor, Extra Trees Regressor, and HistGradientBoosting Regressor), validation metrics (RMSLE as primary metric, MAE in USD, R2), actual vs. predicted parity scatter plots, residual distribution with documented 1st-99th percentile trimming, feature importance analysis, real-time interactive revenue prediction form with range and future date warnings, and Kaggle test-set submission generator (`submission.csv`).

---

## Directory Structure

```text
IDSE_Project/
├── app.py                     # Streamlit application entry point and sidebar dispatcher
├── requirements.txt           # Environment dependencies
├── README.md                  # Project documentation
├── data/                      # Dataset repository (train.csv, test.csv, sample_submission.csv)
├── tmdb-box-office-prediction/ # Official Kaggle competition dataset files
├── notebooks/
│   └── exploratory_analysis.ipynb # Complete 22-section EDA and ML Jupyter Notebook
├── src/
│   ├── __init__.py            # Package exports
│   ├── data_loader.py         # Anchored dataset discovery, safe JSON parsing, and century date parsing
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

## Dataset Location

The official Kaggle TMDB competition datasets are located in either:
- `data/` (`train.csv`, `test.csv`, `sample_submission.csv`)
- `tmdb-box-office-prediction/` (`train.csv`, `test.csv`, `sample_submission.csv`)

The data loader automatically resolves dataset files anchored to the project root directory across environments.

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

### Opening and Executing the Notebook

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Running the Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Machine Learning Formulation

- **Problem Type**: Supervised Regression
- **Target Variable**: `revenue` (Worldwide Box Office Revenue in USD)
- **Target Transformation**: Models fit on `log1p(revenue)` to stabilize exponential variance across skewed theatrical returns; predictions converted with `expm1` and clipped to ensure non-negative revenue estimates.
- **Evaluation Metrics**:
  - **RMSLE** (Root Mean Squared Logarithmic Error) - Primary competition metric.
  - **MAE** (Mean Absolute Error in USD).
  - **R2** (Coefficient of Determination).
- **Leakage Prevention**: Strictly excludes post-release attributes (`revenue`, `popularity`, `vote_average`, `vote_count`) and raw identifiers (`id`, `imdb_id`, `poster_path`, `homepage`). Preprocessing and imputation statistics are derived strictly from the training partition and reused for validation and testing.
