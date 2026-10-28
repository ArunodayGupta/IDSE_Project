# 🎬 Netflix Content Strategy & Catalog EDA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-red.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

An executive-level **Exploratory Data Analysis (EDA) and Business Intelligence web application** investigating Netflix's transition from a studio DVD/digital syndication model into a vertically integrated, global streaming hegemon.

Built with a modular Python pipeline, interactive **Plotly** visualizations, and an executive **Streamlit** dashboard powered by the official 8,800+ title Kaggle Netflix catalog dataset.

---

## 📌 Executive Summary & Core Business Findings

| Strategic Question | Executive Takeaway | Data Evidence |
| :--- | :--- | :--- |
| **1. The Studio Defense Pivot** | Early Netflix depended on legacy studio licensing. In anticipation of Disney+, Peacock, and HBO Max withdrawing back-catalogs, Netflix aggressively transitioned capital into **in-house Originals** and same-year exclusive releases. | Pre-2016 acquisitions were dominated by back-catalogs (**>3 yr lag: 55%+**). By 2021+, same-year releases (**lag = 0**) surged to **over 50%** of all additions. |
| **2. Feature Film Sweet Spot** | Film runtimes are strictly clustered around **90–110 minutes**. Streaming algorithms maximize completion rates (reducing drop-off) and optimize ad-insertion inventory. | Median movie runtime is **98–104 minutes**, with sharp drop-offs for runtimes exceeding 130 minutes. |
| **3. The TV "Season 2 Cliff"** | Series face an aggressive culling rate due to Netflix's cost-plus talent agreements and declining marginal subscriber acquisition per season. | **>60%** of TV series conclude after Season 1; **>80%** do not survive past Season 2. |
| **4. Global Localization Dynamics** | Expansion into India, UK, and South Korea requires customized content rating distributions. Western markets skew mature; emerging markets skew family/teen. | US/UK catalogs skew heavily toward **Mature / Adults (TV-MA, R)**, while India features significantly higher **Family / Teens (TV-14, TV-PG)** co-viewing titles. |
| **5. The "Missing Director" Hypothesis** | High missing director rates in streaming catalogs are **structural** to the medium of episodic TV, rather than poor data collection. | TV Shows exhibit **~74% null director rates** (due to showrunners and rotating episodic directors), whereas Movies exhibit **<12% null rates**. |

---

## 🏗️ System Architecture & Engineering Rigor

```text
netflix-content-strategy-eda/
├── data/
│   ├── .gitkeep
│   └── netflix_titles.csv          # Official Kaggle dataset (8,800+ titles)
├── src/
│   ├── __init__.py                 # Clean package exports
│   ├── data_generator.py           # Robust dataset presence & path verification
│   ├── data_cleaner.py             # Legacy data cleaning & missing value audit
│   ├── data_explorer.py            # Phase 1: Core profiling, schema, missingness & filtering engine
│   ├── explorer_ui.py              # Phase 1: Streamlit UI components & visualizer
│   ├── cleaning_lab.py             # Phase 2: Transformation engine, registry, What Changed?, JSON replay
│   ├── cleaning_lab_ui.py          # Phase 2: Cleaning & Feature Engineering interactive UI
│   ├── model_lab.py                # Phase 3: Predictive modeling engine & memory-safe recommender service
│   ├── model_lab_ui.py             # Phase 3: Model Lab UI (Predictive Studio & Recommender Arena)
│   ├── feature_engineer.py         # Licensing lag, audience tiering, multi-label exploders & metadata soup
│   ├── model_trainer.py            # Legacy TF-IDF & Cosine Similarity NLP Content Recommender
│   └── visualizations.py           # Plotly chart engine with bespoke corporate styling
├── tests/
│   ├── __init__.py
│   ├── test_data_explorer.py       # Automated unit tests for Phase 1 DataLab Explorer (27 tests)
│   ├── test_cleaning_lab.py        # Automated unit tests for Phase 2 Clean & Transform (25 tests)
│   └── test_model_lab.py           # Automated unit tests for Phase 3 Model Lab (19 tests)
├── notebooks/
│   └── exploratory_analysis.ipynb  # Step-by-step narrative Jupyter notebook
├── app.py                          # Streamlit executive dashboard & DataLab workspace
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Strategic documentation & portfolio case study
```

### The 5 Pipeline Phases

```mermaid
flowchart LR
    A["Raw Data Ingestion<br>(data/netflix_titles.csv)"] --> B["Data Hygiene & Cleaning<br>(Date/Duration Regex, Imputation)"]
    B --> C["Feature Engineering<br>(Lag, Audience Tiers, Metadata Soup)"]
    C --> D["Multi-Label Exploder & Visual Engine<br>(Country/Genre Arrays, Plotly Dark)"]
    D --> E["NLP ML Recommender Engine<br>(TF-IDF Vectorizer + Cosine Sim)"]
    E --> F["Streamlit Executive UI<br>(Slicers, KPIs, Deep Dives, AI Cards)"]
```

1. **Phase 1: Ingestion & Dataset Verification (`src/data_generator.py`)**
   - Validates that `data/netflix_titles.csv` is present in the `data/` directory.
   - Raises informative `FileNotFoundError` if missing, guaranteeing reproducible pipeline execution.

2. **Phase 2: Data Hygiene & Parsing Pipeline (`src/data_cleaner.py`)**
   - Generates a quantitative **Missing Value Audit Ledger** before and after cleaning.
   - Tests and quantifies the **Missing Director Hypothesis** (TV vs. Movie null comparisons).
   - Standardizes `date_added` into parsed datetime and extracts `year_added`, `month_added`, and `month_name`.
   - Parses `duration` via regex into `duration_min` (Movies only) and `seasons_count` (TV Shows only).

3. **Phase 3: Strategic Feature Engineering (`src/feature_engineer.py`)**
   - **Acquisition Lag (`acquisition_lag`):** Calculated as `year_added - release_year` (clipped at min 0).
   - **Content Origin Proxy (`content_origin_proxy`):** Segmented into `Same-Year (Original/Exclusive Proxy)`, `Recent Release (1-3 Yrs)`, and `Back-Catalog (>3 Yrs)`.
   - **Audience Maturity Tier (`audience_tier`):** Maps 10+ TV/MPAA ratings into 3 clean business tiers: *Kids*, *Family / Teens*, and *Mature / Adults*.
   - **Multi-Label Relational Exploder (`explode_column`):** Splits comma-separated strings (`country`, `listed_in`) to enable granular geographic and genre co-occurrence counting without mutating or duplicating rows in the parent DataFrame.

4. **Phase 4: Plotly Visualization Engine (`src/visualizations.py`)**
   - Bespoke Netflix Dark Design System (`#E50914` primary red, `#141414` background, translucent hover cards).
   - Five interactive figures:
     1. `plot_catalog_growth`: Catalog volume velocity and TV vs. Movie pivot.
     2. `plot_acquisition_lag_trend`: 100% normalized stack of the studio defense pivot.
     3. `plot_top_countries_audience`: Horizontal stacked bar of top producing nations by audience tier.
     4. `plot_genre_cooccurrence`: 2D pairwise correlation heatmap of co-occurring genres.
     5. `plot_duration_distributions`: Side-by-side subplots of movie runtime sweet spots and TV season drop-offs.
     6. `plot_missing_directors_comparison`: Visual proof of episodic TV director dispersion.

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/netflix-content-strategy-eda.git
cd netflix-content-strategy-eda

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Executive Dashboard

```bash
streamlit run app.py
```
*Or using the Python module:*
```bash
python -m streamlit run app.py
```
The interactive dashboard will open automatically in your browser at `http://localhost:8501`.

### 3. (Optional) Run the Narrative Jupyter Notebook

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 📊 Kaggle Dataset Integration

The dashboard permanently connects to the official Kaggle dataset: **[Netflix Movies and TV Shows by Shivam Bansal](https://www.kaggle.com/datasets/shivamb/netflix-shows)**.

The full dataset of **8,807 titles** is placed at `data/netflix_titles.csv`. The data pipeline validates and loads this catalog automatically upon application startup.

---

## 🔬 DataLab — Phase 1: Interactive Dataset Explorer

Phase 1 transforms this project from a static dashboard into an interactive exploratory data science workspace called **DataLab**. Users can dynamically inspect schema quality, investigate column distributions, detect anomalies, filter records, inspect individual rows, and export findings.

### 1. Launching the Application

Run the application from the repository root:
```bash
streamlit run app.py
```
Or via the Python module:
```bash
python -m streamlit run app.py
```
The application will launch at `http://localhost:8501`. Navigate to the primary tab: **`🔬 DataLab: Explore Dataset`**.

### 2. Dataset Source Selection & Loading
The explorer supports two isolated data modes:
* **Option A: Built-in Netflix Catalog (8,807 titles)** — Resolves the verified Kaggle catalog from `data/netflix_titles.csv` or root fallbacks relative to the project root.
* **Option B: Upload Custom CSV** — Allows users to upload their own CSV file via `st.file_uploader`.
  * **Encoding Robustness:** Automatically tries `UTF-8 with BOM (utf-8-sig)`, `UTF-8`, `Latin-1 (ISO-8859-1)`, and `CP1252 (Windows-1252)`.
  * **Edge Case Handling:** Safely handles empty/0-byte files, duplicate column headers, wide datasets (>100 columns), and mixed data types with clear user-facing alerts.
  * **Session Isolation:** Uploaded data remains strictly session-local. Uploading a custom dataset never overwrites files on disk or corrupts the legacy Netflix analytical models.

### 3. Dataset Overview Dashboard & Metrics
When a dataset is loaded, dynamic KPI cards present actual calculations:
* **Row Count:** Displays current active count with indicator if filters are active (e.g. `Filtered from 8,807 total rows`).
* **Columns:** Total column count with logical type breakdown (`Numeric`, `Categorical/Text`, `Datetime`, `Boolean`).
* **Duplicate Rows:** Quantifies identical duplicate records across all columns.
* **Missing Cells:** Total missing cells and overall dataset missingness percentage.
* **Complete Rows:** Count and percentage of records containing zero missing values.
* **Memory Usage:** Precise deep memory consumption formatted in KB/MB.

### 4. Interactive Exploration Subtabs

1. **📋 Column Schema & Data Dictionary:**
   * Interactive table detailing Column Name, Logical Type, Pandas DataType, Non-Null Count, Missing Count, Missing %, Distinct Values, and representative non-null Sample Values.
   * Search box to filter columns by name for wide datasets.
   * Direct download button for the column profiling report (`CSV`).

2. **🔍 Missing-Value Audit & Visualizer:**
   * **Missingness Convention:** Includes a toggle to *"Treat empty or whitespace-only strings as missing"* (enabled by default). This treats `""`, `"   "`, and standard nulls consistently without mutating source data.
   * **Missing Value Rankings:** Columns ordered descending by missing count and percentage with severity badges (`Critical >50%`, `High >20%`, `Moderate >0%`, `Zero Missing`).
   * **Missingness Bar Chart:** Interactive Plotly bar chart visualizing missing percentages per feature.
   * **Sparsity Matrix Heatmap:** 2D heatmap showing missing (red) vs. present (blue) cells across the dataset.
   * **Inspect Missing Records:** Drill-down table to isolate records with missing values in any selected column.

3. **📊 Column Distributions & Deep Dive:**
   * **Numeric Columns:** Non-missing count, Mean, Median, Std Dev, Min, Max, Quartiles (Q1, Q3), IQR, Skewness, Zeros count, and Outlier detection via standard `1.5 * IQR` rule. Configurable histogram bins slider and outlier box plot.
   * **Categorical/Text Columns:** Unique count, Top Category share %, configurable Top-N category slider (5–50), interactive horizontal frequency bar chart, frequency data table, and interactive record inspector for selected categories.
   * **Datetime Columns:** Valid dates count, earliest date, latest date, time span in days, and yearly timeline frequency chart.
   * **Non-Destructive Date Preview:** Preview date parsing on text columns without mutating original series.

4. **🔎 Filter & Search Workspace:**
   * **Deterministic AND Logic:** Multiple filters combine predictably with AND logic.
   * **Global Search:** Case-insensitive substring query across text fields.
   * **Numeric Range Filtering:** Min-max range sliders for any numeric column.
   * **Categorical Selection:** Multi-select filtering with search support.
   * **Text Substring:** Specific column substring queries.
   * **Missingness Filtering:** Filter by rows that are missing or non-missing in a selected column, or rows with any missing cell.
   * **Reset Action:** `🔄 Reset All Filters` button restores the full dataset view without resetting dataset source selection.

5. **📑 Record Browser & Inspector:**
   * Paginated record table with configurable page size (25, 50, 100 rows).
   * Column visibility multi-selector to choose displayed attributes.
   * Sort by any column in Ascending or Descending order.
   * **Individual Record Inspector:** Select any record from the displayed page to inspect every attribute in full detail with formatted nulls (`[Missing / Null]`).

### 5. Export Functionality
* **Filtered Data Export (CSV):** Downloads all currently filtered records (not just the visible page) with original column order, excluding internal helper columns (`index=False`).
* **Data Profile Report Export (CSV):** Downloads the complete column schema data dictionary.

### 6. Phase 2: 🧹 DataLab: Clean & Transform Lab
Phase 2 transforms DataLab into an interactive data preparation, cleaning, and feature engineering lab. It bridges raw exploratory inspection with downstream modeling by letting users clean data and engineer features step-by-step.

#### 1. Seamless Shared Dataset State
* The source dataset selected in Phase 1 (Built-in Netflix or Uploaded CSV) is automatically shared with Phase 2 without requiring re-uploading.
* Cryptographic SHA-256 source identity invalidation ensures that switching sources immediately refreshes the cleaning workspace and resets stale transformations.
* The original raw source remains **completely immutable**; all cleaning actions operate on an isolated working DataFrame.

#### 2. Initial Data Quality Audit
* Diagnostic summary displaying row/column dimensions, total missing cells & percentage, duplicate count, constant columns, 100% missing columns, blank/whitespace-only text columns, and parseable date candidates.

#### 3. Available Cleaning Operations
* **Missing-Value Strategies:**
  - Drop rows missing values across all or selected columns (`any` / `all`).
  - Drop unneeded or high-null columns.
  - Impute numeric columns with **Mean** or **Median** (with zero usable number safety).
  - Impute categorical columns with **Mode** (most frequent valid category).
  - Fill missing cells with a user-provided **Constant Value**.
  - Convert blank or whitespace-only text cells to proper `np.nan` missing values.
* **Duplicate Management:**
  - Choose duplicate definitions across all columns or selected subsets.
  - Apply policies: **Keep First Occurrence**, **Keep Last Occurrence**, or **Remove All Members**.
* **Type Conversion & Text Normalization:**
  - Explicit numeric and datetime parsing with non-destructive preview and coercion error handling.
  - Text normalization: leading/trailing whitespace trimming, lowercase, uppercase, and title-case.

#### 4. Feature Engineering Playground
Pure allowlisted functional transformations (strictly **NO** `eval()` or `exec()` for security):
* **Numeric Arithmetic:** Addition, subtraction, multiplication, and division across two columns or scalar values. Includes automatic divide-by-zero protection (coerces to `np.nan` instead of crashing or generating `inf`).
* **Mathematical Transforms:** Absolute value (`abs`), natural logarithm (`log`), and `log1p` with automatic invalid non-positive input safety.
* **Binning:** Quantile-based or uniform interval binning into user-selected tiers.
* **Datetime Decomposition & Delays:** Extract year, month, day, day of week, quarter, or calculate time differences between date fields in days.
* **Categorical Mapping & Indicators:** User-defined category dictionary mapping, binary indicators (0/1), and manageable one-hot encoding.
* **Text Features:** Character length and word count extraction.
#### 5. Phase 2 Quality Repairs & Hardening (Part A)
* **Explicit Boolean Conversion:** Robust parsing mapping `true`/`1`/`yes`/`y` to `True` and `false`/`0`/`no`/`n` to `False`. Rejects arbitrary non-empty strings and handles actual booleans, 0/1 numerics, and nullable `boolean` dtypes with clear invalid-count reporting.
* **Preservation of Missing Strings:** String type conversion preserves null representations as nullable missing entries (`pd.NA`) rather than coercing them into literal `"nan"` or `"None"` strings.
* **Constant-Fill Key-Presence:** Replaced truthiness checks with explicit key-presence checks (`"constant_value" in params`), allowing valid falsy constant imputations (`0`, `False`, `""`). Automatically adds new categories to `CategoricalDtype` before imputation to eliminate pandas assignment failures.
* **Missing Value Preservation Across Feature Engineering:** Operations (`binning`, `category_map`, `binary_indicator`, `one_hot`, `text_length`, `word_count`, `datetime_extract`, `date_diff`) preserve `np.nan` and `pd.NA` without stringifying them into fake `"nan"` category labels.
* **Strengthened Column Collision Protection:** Pre-calculates intended output column names before transformations, validates collisions across all generators (including normalized collisions like `"A B"` vs `"A_B"`), and rejects collisions unless deliberate `overwrite=True` is provided.
* **Enhanced "What Changed?" Engine:** Quantifies the exact number of modified cells within retained rows, provides representative before-and-after values for altered cells, isolates removed records for user inspection, and details per-column missingness deltas.
* **Transactional JSON Pipeline Replay:** Fully reproducible pipeline import/replay engine that dry-runs configurations against the selected dataset, validates against an allowlisted registry, and applies transformations transactionally with automatic rollback on error.
* **Inclusive Date-Filter Semantics:** Phase 1 date filters enforce an inclusive start date and an exclusive next-day midnight boundary (`end_date + 1 day`), correctly retaining timestamps throughout the selected end date without timezone mismatch errors.
* **Source-State Contract:** Internal helpers (`get_datalab_source_dataset()`, `get_datalab_working_dataset()`) ensure downstream workspaces can access either the immutable source or the active working copy.

---

### 🤖 DataLab Phase 3: Model Lab — ML Experiment Arena (Part B)

The Model Lab connects data exploration and feature engineering into an empirical machine learning studio with two primary modes:

#### Mode 1: Predictive Modeling Studio
* **Dataset Selection:** Train on the original raw source dataset or the active Phase 2 transformed working copy, with automatic stale-model detection when datasets or features change.
* **Task Detection & Target Validation:** Automatically suggests Classification or Regression based on target cardinality and data types. Handles binary and multiclass classification, target missingness exclusion (with transparent reporting), and screens out long unstructured free-text columns from tabular estimators with explanatory guidance.
* **Leakage-Free Preprocessing Pipeline:** Uses Scikit-Learn `ColumnTransformer` and `Pipeline` fitted **strictly on the training split** ($X_{train}$):
  - Numeric columns: `SimpleImputer(strategy="median")` + `StandardScaler()`.
  - Categorical columns: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
* **Candidate Model Suite:**
  - *Classification:* Baseline `DummyClassifier(strategy="most_frequent")`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`.
  - *Regression:* Baseline `DummyRegressor(strategy="mean")`, `Ridge` / Linear Regression, `DecisionTreeRegressor`, `RandomForestRegressor`.
* **Empirical Evaluation Metrics:**
  - *Classification:* Accuracy, Balanced Accuracy, Macro F1, Weighted F1, Precision, Recall, interactive Plotly Confusion Matrix, Per-Class Classification Report, and ROC-AUC (when probabilities are supported).
  - *Regression:* MAE, RMSE, $R^2$, Residual Distribution plot, and Predicted vs. Actual scatter view.
* **Cross-Validation:** Optional $K$-fold / Stratified $K$-fold cross-validation executed solely on the training partition with preprocessing fitted inside each fold.
* **Feature Importance & Interpretability:** Extracts tree-based Mean Decrease in Impurity (MDI) importances and linear regression coefficients mapped back to readable feature names (including one-hot categories).
* **Model Error Analysis:** Test-set error inspection table showing top misclassified instances with prediction confidence (classification) or largest absolute residuals (regression).

#### Mode 2: Recommender Experiment Arena
* **Memory-Safe Sparse Cosine Similarity:** Refactored from dense $N \times N$ matrix allocation to on-demand query dot product (`cosine_similarity(query_vec, catalog_matrix)`). Eliminates hundreds of megabytes of redundant memory while maintaining 100% backward compatibility with legacy recommender interfaces.
* **Multi-Field TF-IDF & Weighting:** Users can select individual metadata fields (`genres`, `director`, `cast`, `description`, etc.), configure hyperparameters (`max_features`, n-gram ranges, sublinear TF, stop words), and apply granular field weights (including complete field ablation at weight `0.0`).
* **A/B Comparison on Same Query:** Run two distinct recommender configurations against the identical query title to visualize:
  - Jaccard Top-$K$ Overlap %.
  - Shared recommendations, entering items, and leaving items.
  - Per-item rank shifts and similarity score changes.
* **Honest Relevance Feedback (No Fabricated Ground Truth):** In-session interactive relevance feedback (thumbs up / thumbs down). Computes judged **Precision@K** and **NDCG@K** based solely on user-evaluated items, explicitly disclosing judged vs. unjudged counts without making false assumptions about unrated catalog items.

#### Subtab 3: Saved Experiment Tracker & Export Hub
* **Session Run Logging:** Automatically logs experiment configurations, hyperparameters, dataset identities, execution runtimes, and metrics.
* **Export Hub:** Download experiment scorecards as CSV or export model/recommender configurations as validated JSON.

---

### 🧪 Running Automated Tests
Run the comprehensive automated unit test suite from the project root:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
**All 71 automated tests pass successfully**, covering:
1. `tests/test_data_explorer.py` (27 tests): Dataset SHA-256 identity, linspace sampling across 501–10,000 rows, unified missingness semantics, inclusive date bounds, timezone awareness, pagination, CSV exports.
2. `tests/test_cleaning_lab.py` (25 tests): Explicit boolean mapping, null preservation in string conversion, falsy constant fills (`0`, `False`, `""`), categorical category expansion, feature engineering missing preservation, column collision prevention, What Changed modified/removed tracking, transactional JSON pipeline replay and error rollback.
3. `tests/test_model_lab.py` (19 tests): Heuristic task detection, target-missing row dropping, leakage-free preprocessing pipeline fitting on $X_{train}$, classification candidate models & ROC-AUC, regression candidate models & residuals, cross-validation, feature importance extraction, error analysis, memory-safe sparse recommender similarity, A/B Jaccard comparisons, judged NDCG/Precision relevance feedback, and JSON configuration exports.

---

### ⚡ Performance Constraints & Limitations
- **Tabular Modeling:** Focuses on structured numeric and categorical features; long free-text columns (average length > 60 chars) are screened out of tabular estimators and directed toward the Recommender Experiment Arena.
- **Recommender Evaluation:** Real user preference ground truth is not artificially manufactured; ranking metrics (Precision@K, NDCG@K) require user relevance judgments and explicitly display unjudged item counts.
- **In-Memory Scalability:** Designed for interactive single-node exploration and modeling up to ~100,000 records. Estimator training utilizes Scikit-Learn with bounded tree depths and tree counts to maintain sub-second interactive response times.

---

## 📊 Dashboard Tour

### Tab 0: 🔬 DataLab: Explore Dataset
Interactive dataset exploration workspace: schema profiling, missingness visualizers, column distributions, multi-condition filtering, date filtering, high-cardinality search, record inspection, and CSV exports.

### Tab 1: 🧹 DataLab: Clean & Transform
First-class interactive data cleaning and feature engineering workspace: automated quality audit, missingness handling, deduplication, text normalization, allowlisted feature engineering playground, strategic Netflix recipes, What Changed? comparison card, step audit ledger, undo/reset state management, JSON pipeline import/replay, and export hub.

### Tab 2: 🤖 DataLab: Model Lab
Empirical ML Experiment Arena:
- **Mode 1 — Predictive Modeling Studio:** Supervised classification and regression on raw source or transformed working copy, baseline comparisons, cross-validation, Plotly confusion matrix / residual distribution, feature importances, and error inspection.
- **Mode 2 — Recommender Experiment Arena:** Multi-field TF-IDF with field weighting and ablation, memory-safe on-demand sparse ranking, A/B query testing with Jaccard overlap, and user relevance feedback metrics.
- **Experiment Tracker:** In-session run comparison, CSV export, and JSON configuration serialization.

### Tab 3: 📈 Executive Strategy & Growth
- **Catalog Expansion Velocity:** Tracks yearly title additions split by Movies vs. TV Shows, revealing the post-2016 inflection point.
- **The Studio Defense Pivot:** 100% normalized view showing the transition from studio back-catalogs to original/same-year releases.
- **Annual Ingestion Ledger:** Quantitative summary table of volume, TV share %, and Originals proxy %.

### Tab 4: 🌍 Global Footprint & Ratings
- **Production Powerhouses:** Top 14 producing nations segmented by audience maturity tiers (*Kids*, *Family / Teens*, *Mature / Adults*).
- **International Content Strategy Matrix:** Highlights localization differences between Western markets and emerging high-growth regions like India and South Korea.

### Tab 5: 🎭 Genre & Format Mechanics
- **Runtime Optimization:** Movie runtime distribution showing the median and the 90–110 min sweet spot.
- **The Season 2 Cliff:** TV season attrition rates illustrating the financial dynamics behind series cancellations.
- **Genre Co-Occurrence Heatmap:** 2D matrix identifying which content categories are most frequently packaged together.

### Tab 6: 🛠️ EDA Data Hygiene & Engineering Log
- **Hygiene KPI Scorecard:** Initial row count, final cleaned count, and dropped anomaly rate (<1.5%).
- **Missing Director Hypothesis Deep Dive:** Interactive visual and statistical proof explaining episodic TV director attribution.
- **Missing Value Audit Ledger:** Comprehensive before-and-after table tracking null counts and percentages across all features.
- **Cleaned Data Explorer & CSV Export:** Interactive table with one-click export for downstream modeling.

### Tab 7: 🎯 Legacy Content Recommendation Engine
- **NLP Metadata Soup Construction:** Blends lowercased titles, sanitized director/cast name tokens (`Tom Hanks` -> `tomhanks`), genre categories, and descriptions.
- **On-Demand Memory-Safe Sparse Similarity:** Optimized Scikit-Learn vectorizer generating semantic embeddings without full $N \times N$ matrix memory overhead.
- **Interactive Title Search & Slicer Integration:** Instant query lookup across the active catalog slice with dedicated target spotlight.
- **Bespoke Visual Recommendation Cards:** High-end glassmorphism containers showcasing similarity match %, format, rating, metadata tokens, and narrative synopses.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing & Manipulation:** `pandas>=2.0.0`, `numpy>=1.24.0`
* **Machine Learning & NLP:** `scikit-learn>=1.3.0` (ColumnTransformer, Pipelines, Estimators, TF-IDF Vectorizer, Cosine Similarity)
* **Interactive Visualization:** `plotly>=5.18.0`
* **Web Application Framework:** `streamlit>=1.30.0`
* **Statistical Modeling & Distribution:** `scipy>=1.11.0`
---
