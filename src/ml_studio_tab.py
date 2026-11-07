"""
ml_studio_tab.py
Section C: ML Studio for TMDB Worldwide Movie Box Office Revenue Prediction.
Consolidated, single-page dashboard containing:
1. Target leakage prevention & pre-release feature configuration.
2. Model training and comparison (Dummy baseline, Random Forest, Extra Trees, HistGradientBoosting).
3. Model evaluation (RMSLE primary, MAE in USD, R2, actual vs predicted, residuals, feature importance).
4. Interactive pre-release movie revenue prediction form with currency formatting & range warnings.
5. Kaggle test-set submission generator (preserves test IDs, produces submission.csv).
Strictly adheres to professional styling with zero emojis.
"""

from typing import Dict, Any, Optional
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.data_loader import load_tmdb_test_data, get_dataset_availability
from src.feature_engineering import (
    engineer_movie_features,
    prepare_single_movie_features,
    CORE_GENRES,
    TOP_LANGUAGES,
)
from src.model_trainer import (
    train_movie_regressor,
    predict_single_movie,
    generate_kaggle_submission,
    MODEL_REGISTRY,
)
from src.visualizations import (
    plot_actual_vs_predicted,
    plot_residuals_distribution,
    plot_feature_importance,
    format_currency_compact,
)
from src.ui_theme import (
    render_page_header,
    render_section_header,
    render_kpi_row,
    render_metric_card,
    render_callout,
    render_badge,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_VIOLET,
    COLOR_SUCCESS,
    COLOR_WARNING,
)


@st.cache_data(show_spinner=False)
def get_cached_features_and_splits(
    raw_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Establishes a leak-free train/validation partition.

    Leakage Prevention Workflow:
    1. Filter labeled records with strictly positive revenue.
    2. Split raw records into training and validation sets BEFORE feature engineering.
    3. Calculate imputation medians and reference statistics strictly from the training partition.
    4. Transform validation records using only the training-derived statistics.
    """
    if "revenue" not in raw_df.columns:
        raise ValueError("Training dataset must contain the target column 'revenue'.")

    # 1. Isolate labeled records with positive revenue
    valid_revenue_mask = pd.to_numeric(raw_df["revenue"], errors="coerce") > 0
    labeled_raw = raw_df.loc[valid_revenue_mask].copy()

    # 2. Split raw records prior to any feature transformation
    raw_train, raw_val = train_test_split(
        labeled_raw, test_size=test_size, random_state=random_state
    )

    # 3. Engineer features on training partition; compute stats on training only
    X_train, y_train, stats = engineer_movie_features(raw_train, is_train=True, stats=None)

    # 4. Transform validation partition reusing training statistics
    X_val, y_val, _ = engineer_movie_features(raw_val, is_train=True, stats=stats)

    return {
        "stats": stats,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


def render_ml_studio_section(raw_df: pd.DataFrame) -> None:
    """Renders the consolidated ML Studio dashboard."""
    render_page_header(
        title="ML Studio",
        subtitle="Pre-release movie metadata modeling, evaluation, and worldwide revenue forecasting",
        badge="Supervised Regression",
        badge_type="info"
    )

    # Session State and Dataset Invalidation Management
    dataset_fingerprint = f"{len(raw_df)}_{raw_df.shape[1]}_{id(raw_df)}"
    if st.session_state.get("ml_active_dataset_fp") != dataset_fingerprint:
        st.session_state["ml_studio_models"] = {}
        st.session_state["ml_active_dataset_fp"] = dataset_fingerprint

    # Prepare features and train/validation split using leak-free pipeline
    split_bundle = get_cached_features_and_splits(raw_df, test_size=0.2, random_state=42)
    X_train = split_bundle["X_train"]
    y_train = split_bundle["y_train"]
    X_val = split_bundle["X_val"]
    y_val = split_bundle["y_val"]
    stats = split_bundle["stats"]

    # Target leakage prevention notice
    render_callout(
        title="Target Leakage Prevention Contract",
        body=(
            "Predictive features are restricted strictly to pre-release metadata: production budget, "
            "genres, runtime, release date, original language, production countries, production companies, "
            "and cast/crew sizes. Post-release metrics (revenue, popularity, vote counts, and vote averages) "
            "are completely excluded to ensure model validity. Preprocessing statistics are calculated "
            "exclusively from training records."
        ),
        callout_type="info"
    )

    # -------------------------------------------------------------------------
    # 1. Model Training & Comparison Controls
    # -------------------------------------------------------------------------
    render_section_header("Model Training & Comparison", "Configure hyperparameters and compare algorithms on validation data")

    t_col1, t_col2 = st.columns([1, 1], gap="medium")

    with t_col1:
        model_options = {
            "dummy": "Dummy Regressor (Baseline)",
            "random_forest": "Random Forest Regressor",
            "extra_trees": "Extra Trees Regressor",
            "hist_gradient_boosting": "HistGradientBoosting Regressor",
        }
        selected_model_key = st.selectbox(
            "Select Algorithm",
            options=list(model_options.keys()),
            format_func=lambda k: model_options[k],
            index=1,
            key="ml_algo_select"
        )

        st.caption(MODEL_REGISTRY[selected_model_key]["description"])

    with t_col2:
        # Dynamic Hyperparameter Controls
        hyperparams: Dict[str, Any] = {}
        if selected_model_key in ("random_forest", "extra_trees"):
            p1, p2, p3 = st.columns(3)
            with p1:
                hyperparams["n_estimators"] = st.slider("Estimators", 20, 200, 100, step=10, key=f"{selected_model_key}_nest")
            with p2:
                hyperparams["max_depth"] = st.slider("Max Depth", 4, 25, 12, step=1, key=f"{selected_model_key}_depth")
            with p3:
                hyperparams["min_samples_split"] = st.slider("Min Split", 2, 10, 4, step=1, key=f"{selected_model_key}_split")

        elif selected_model_key == "hist_gradient_boosting":
            p1, p2, p3 = st.columns(3)
            with p1:
                hyperparams["max_iter"] = st.slider("Max Iterations", 30, 250, 100, step=10, key="hgb_iter")
            with p2:
                hyperparams["max_depth"] = st.slider("Max Depth", 3, 12, 6, step=1, key="hgb_depth")
            with p3:
                hyperparams["learning_rate"] = st.slider("Learning Rate", 0.01, 0.30, 0.08, step=0.01, key="hgb_lr")

        elif selected_model_key == "dummy":
            st.info("Dummy baseline predicts median training set revenue with zero tunable hyperparameters.")

    # Train Model button
    if st.button("Train Model", type="primary", key="btn_train_model"):
        with st.spinner(f"Fitting {model_options[selected_model_key]} on log1p(revenue)..."):
            res = train_movie_regressor(
                model_key=selected_model_key,
                hyperparams=hyperparams,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                random_state=42
            )
            st.session_state["ml_studio_models"][selected_model_key] = res
            st.session_state["ml_active_model_key"] = selected_model_key
            st.success(f"Trained {res['model_name']} in {res['fit_duration_sec']}s. Validation RMSLE: {res['rmsle']:.4f}")

    # Auto-train default models on first load if registry is empty
    if not st.session_state.get("ml_studio_models"):
        with st.spinner("Initializing baseline and Random Forest models..."):
            res_dummy = train_movie_regressor("dummy", {}, X_train, y_train, X_val, y_val)
            st.session_state["ml_studio_models"]["dummy"] = res_dummy

            res_rf = train_movie_regressor("random_forest", {"n_estimators": 100, "max_depth": 12}, X_train, y_train, X_val, y_val)
            st.session_state["ml_studio_models"]["random_forest"] = res_rf
            st.session_state["ml_active_model_key"] = "random_forest"

    trained_registry = st.session_state["ml_studio_models"]

    # Model Comparison Table (Displays only actually trained models)
    if trained_registry:
        table_rows = []
        best_rmsle = min(m["rmsle"] for m in trained_registry.values())

        for k, m in trained_registry.items():
            is_best = (m["rmsle"] == best_rmsle)
            table_rows.append({
                "Algorithm": m["model_name"] + (" (Lowest RMSLE)" if is_best else ""),
                "RMSLE (Primary)": f"{m['rmsle']:.4f}",
                "MAE (USD)": f"${m['mae']:,.0f}",
                "R2 Score": f"{m['r2']:.4f}",
                "Fit Time": f"{m['fit_duration_sec']:.2f}s",
                "_key": k,
                "_rmsle_raw": m["rmsle"]
            })

        df_comp = pd.DataFrame(table_rows).sort_values(by="_rmsle_raw").reset_index(drop=True)
        display_comp = df_comp[["Algorithm", "RMSLE (Primary)", "MAE (USD)", "R2 Score", "Fit Time"]]
        st.markdown("**Session Model Comparison:**")
        st.dataframe(display_comp, use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Model Evaluation & Diagnostics
    # -------------------------------------------------------------------------
    render_section_header("Model Evaluation & Diagnostics", "Validation metrics, parity analysis, residuals, and feature importance")

    active_key = st.session_state.get("ml_active_model_key", "random_forest")
    # Allow user to toggle active model for inspection
    eval_model_key = st.selectbox(
        "Inspect Model Results",
        options=list(trained_registry.keys()),
        format_func=lambda k: trained_registry[k]["model_name"],
        index=list(trained_registry.keys()).index(active_key) if active_key in trained_registry else 0,
        key="ml_inspect_model_select"
    )

    active_res = trained_registry[eval_model_key]

    # Metric KPI Cards
    eval_kpis = [
        {
            "title": "Selected Model",
            "value": active_res["model_name"],
            "subtext": "Active Evaluation",
            "accent_color": COLOR_CYAN
        },
        {
            "title": "Validation RMSLE",
            "value": f"{active_res['rmsle']:.4f}",
            "subtext": "Primary Evaluation Metric (Lower is Better)",
            "accent_color": COLOR_SUCCESS if active_res['rmsle'] < 2.5 else COLOR_WARNING
        },
        {
            "title": "Validation MAE",
            "value": format_currency_compact(active_res["mae"]),
            "subtext": f"Exact: ${active_res['mae']:,.0f}",
            "accent_color": COLOR_BLUE
        },
        {
            "title": "Coefficient of Determination (R2)",
            "value": f"{active_res['r2']:.4f}",
            "subtext": "Variance Explained on Validation Partition",
            "accent_color": COLOR_VIOLET
        }
    ]
    render_kpi_row(eval_kpis)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Evaluation Charts: Actual vs Predicted & Residuals
    e_col1, e_col2 = st.columns([1, 1], gap="medium")

    with e_col1:
        fig_scatter = plot_actual_vs_predicted(
            active_res["y_val_true"],
            active_res["y_val_pred"],
            model_name=active_res["model_name"]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Points closer to the diagonal reference line indicate agreement between predicted and actual revenue.")

    with e_col2:
        fig_res = plot_residuals_distribution(active_res["residuals"])
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("Residual distribution shows prediction errors (actual minus predicted revenue) in millions USD.")

    # Feature Importance Chart (Handled safely for models without feature importances)
    if not active_res["feature_importances"].empty:
        fig_imp = plot_feature_importance(active_res["feature_importances"], top_n=15)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Relative feature importance values derived from tree splits in the fitted model.")
    else:
        st.info(f"Feature importances are not directly available for {active_res['model_name']}.")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Interactive Revenue Prediction
    # -------------------------------------------------------------------------
    render_section_header("Interactive Revenue Prediction", "Estimate worldwide box-office receipts for an upcoming film")

    st.markdown("Enter pre-release specifications to generate a real-time revenue estimate using the selected model:")

    with st.form("form_interactive_prediction"):
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            inp_budget = st.number_input(
                "Production Budget (USD)",
                min_value=0,
                max_value=500000000,
                value=40000000,
                step=1000000,
                help="Enter 0 if budget is unrecorded or indie."
            )
            inp_runtime = st.number_input(
                "Runtime (Minutes)",
                min_value=30,
                max_value=360,
                value=115,
                step=5
            )
            inp_date = st.date_input(
                "Release Date",
                value=datetime.date(2025, 7, 18)
            )

        with p_col2:
            core_genre_names = [g[1] for g in CORE_GENRES]
            inp_genres = st.multiselect(
                "Genres",
                options=core_genre_names,
                default=["Action", "Adventure"]
            )
            lang_options = ["en", "fr", "es", "zh", "de", "hi", "ja", "it", "ko", "ru", "cn", "other"]
            inp_lang = st.selectbox(
                "Original Language",
                options=lang_options,
                index=0
            )
            inp_country = st.selectbox(
                "Primary Production Country",
                options=["US", "GB", "FR", "DE", "CA", "AU", "JP", "CN", "IN", "OTHER"],
                index=0
            )

        with p_col3:
            inp_companies = st.number_input(
                "Number of Production Companies",
                min_value=1,
                max_value=20,
                value=2,
                step=1
            )
            inp_major = st.checkbox("Major Studio Backing", value=True)
            inp_collection = st.checkbox("Belongs to Franchise / Collection", value=False)
            inp_homepage = st.checkbox("Official Website Available", value=False)
            inp_tagline = st.checkbox("Promotional Tagline Available", value=True)
            inp_cast_size = st.number_input(
                "Cast Size",
                min_value=1,
                max_value=100,
                value=20,
                step=1
            )
            inp_crew_size = st.number_input(
                "Crew Size",
                min_value=1,
                max_value=200,
                value=35,
                step=1
            )

        submit_pred = st.form_submit_button("Predict Worldwide Revenue", type="primary")

    if submit_pred:
        movie_dict = {
            "budget": inp_budget,
            "runtime": inp_runtime,
            "release_date": inp_date,
            "genres": inp_genres,
            "original_language": inp_lang,
            "production_country": inp_country,
            "company_count": inp_companies,
            "is_major_studio": inp_major,
            "belongs_to_collection": inp_collection,
            "cast_size": inp_cast_size,
            "crew_size": inp_crew_size,
            "language_count": 1,
            "country_count": 1,
            "has_homepage": inp_homepage,
            "has_tagline": inp_tagline,
        }

        single_feature_df = prepare_single_movie_features(movie_dict, stats)
        pipeline = active_res["pipeline"]
        predicted_usd = predict_single_movie(pipeline, single_feature_df)

        # Range / Outlier validation warnings
        warnings = []
        if inp_date and inp_date.year > 2021:
            warnings.append(
                f"Entered release date ({inp_date}) is beyond historical training observations (up to 2020). "
                "The model is extrapolating into future box-office conditions; forecast carries heightened uncertainty."
            )
        if inp_budget > stats.get("budget_p99", 200000000.0):
            warnings.append(
                f"Entered budget (${inp_budget:,.0f}) exceeds the 99th percentile of training data. "
                "Prediction carries increased variance."
            )
        if inp_runtime < stats.get("runtime_min", 40.0) or inp_runtime > stats.get("runtime_max", 240.0):
            warnings.append(
                f"Entered runtime ({inp_runtime} minutes) is substantially outside typical theatrical parameters."
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        for w in warnings:
            st.warning(w)

        # Display formatted revenue prediction with zero-budget safe multiple
        multiple_html = (
            f"Estimated Gross Multiple: {(predicted_usd / inp_budget):.2f}x production budget"
            if inp_budget > 0
            else "Estimated Gross Multiple: N/A (Unrecorded or zero budget)"
        )

        st.markdown(f"""
            <div style="background: linear-gradient(180deg, rgba(20, 28, 46, 0.95) 0%, rgba(14, 20, 34, 0.95) 100%); border: 1px solid #1E2B44; border-top: 3px solid {COLOR_CYAN}; border-radius: 10px; padding: 22px 26px; margin: 12px 0;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em;">
                    Predicted Worldwide Box Office Revenue
                </div>
                <div style="font-size: 2.8rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.03em; margin: 6px 0;">
                    ${predicted_usd:,.0f} USD
                </div>
                <div style="font-size: 0.92rem; color: #00D2FF; font-weight: 600;">
                    Model: {active_res['model_name']} | Validation RMSLE: {active_res['rmsle']:.4f}
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">
                    {multiple_html}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Kaggle Test-Set Submission Generator
    # -------------------------------------------------------------------------
    render_section_header("Kaggle Test-Set Submission", "Generate predictions for the official TMDB competition test dataset")

    avail = get_dataset_availability()
    test_avail = avail.get("test.csv") is not None

    if test_avail:
        st.markdown("Generate official competition predictions for `test.csv` (4,398 films) preserving original movie IDs:")

        if st.button("Generate Kaggle Submission (submission.csv)", key="btn_gen_submission"):
            with st.spinner(f"Retraining {active_res['model_name']} on 100% of labeled training data and predicting on test.csv..."):
                raw_test, _ = load_tmdb_test_data()
                if raw_test is not None:
                    submission_df = generate_kaggle_submission(
                        model_key=eval_model_key,
                        hyperparams=active_res.get("hyperparams", {}),
                        raw_train_df=raw_df,
                        raw_test_df=raw_test,
                        random_state=42
                    )
                    st.session_state["ml_kaggle_submission"] = submission_df
                    st.success(f"Generated {len(submission_df):,} test predictions successfully.")
                else:
                    st.error("Failed to load test.csv.")

        if "ml_kaggle_submission" in st.session_state:
            sub_df = st.session_state["ml_kaggle_submission"]
            st.markdown("**Submission Preview:**")
            st.dataframe(sub_df.head(8), use_container_width=True)

            sub_buf = io.StringIO()
            sub_df.to_csv(sub_buf, index=False)
            st.download_button(
                label="Download submission.csv",
                data=sub_buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
                key="btn_download_sub_csv"
            )
    else:
        render_callout(
            title="test.csv Not Detected",
            body=(
                "test.csv was not detected in project directories. "
                "Place test.csv into 'data/' or 'tmdb-box-office-prediction/' to enable Kaggle test prediction generation."
            ),
            callout_type="warning"
        )
