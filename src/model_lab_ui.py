"""
model_lab_ui.py
Phase 3 & 4: Model Lab UI.
Consolidates:
1. Predictive Modeling Studio (Classification & Regression with Scikit-learn pipelines).
2. Recommender Experiment Arena (Configurable multi-field TF-IDF, weights, ablation, A/B testing, truthful relevance feedback).
3. Quick Netflix Recommendations (Consolidated from legacy top-level recommendation tab with card presentation).
4. Experiment Tracker & Export Hub (Reliable session tracking, deletion, deduplication, versioned JSON/CSV export).
"""

from typing import Optional, List, Dict, Any
import io
import json
import time
import streamlit as pd_st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .data_context import (
    get_source_dataset,
    get_working_dataset,
    compute_modeling_dataset_identity,
    is_result_stale,
)
from .model_lab import (
    detect_suggested_task,
    validate_modeling_inputs,
    build_preprocessor,
    train_and_evaluate_model,
    extract_feature_importances,
    compute_error_analysis,
    compute_model_config_fingerprint,
    RecommenderArenaService,
    compare_recommender_runs,
    compute_relevance_feedback_metrics,
    export_model_experiment_json,
    export_recommender_experiment_json,
)
from .model_trainer import ContentRecommender
from src.ui_theme import (
    render_page_header,
    render_kpi_row,
    render_callout,
    render_badge,
    apply_figure_theme,
    COLOR_CYAN,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_VIOLET
)


def render_model_lab_tab(netflix_raw_df: Optional[pd.DataFrame] = None) -> None:
    """
    Renders the consolidated Model Lab workspace.
    """
    render_page_header(
        title="Model Lab & Recommender Arena",
        subtitle="Leakage-free predictive modeling, on-demand content recommendations, and empirical metric scorecards.",
        badge="Predictive Studio",
        badge_type="info"
    )

    # ------------------------------------------------------------------
    # Step 1: Shared Dataset State Retrieval (Source vs Working Copy)
    # ------------------------------------------------------------------
    source_df, source_ident, source_name = get_source_dataset()
    working_df, working_ident, working_name, is_transformed, working_state_id = get_working_dataset()

    if source_df is None or source_df.empty:
        if netflix_raw_df is not None and not netflix_raw_df.empty:
            source_df = netflix_raw_df.copy()
            source_name = "Built-in Netflix Catalog"
            source_ident = "netflix_catalog"
            working_df = source_df.copy()
            working_ident = source_ident
            working_name = source_name
            is_transformed = False
            working_state_id = "netflix_catalog__initial"

    if source_df is None:
        st.warning("⚠️ No dataset currently loaded. Please select or upload a dataset in **DataLab Explore**.")
        return

    # Dataset Source Selector
    st.markdown("##### 📁 Input Dataset Selection")
    sel_col1, sel_col2 = st.columns([2, 3])
    with sel_col1:
        data_choice = st.radio(
            "Select Modeling Data Input:",
            options=["Raw Source Snapshot", "Clean & Transform Working Copy"],
            index=1 if is_transformed else 0,
            horizontal=True,
            help="Choose between the original immutable dataset or the transformed working copy from Phase 2."
        )

    is_working_selected = (data_choice == "Clean & Transform Working Copy")
    if is_working_selected and working_df is not None:
        active_df = working_df
        active_dataset_id = f"{source_ident}::working::{working_state_id}"
    else:
        active_df = source_df
        active_dataset_id = f"{source_ident}::raw_source"

    # Defect A1: Explicit zero-row detection
    is_active_empty = (active_df is None or len(active_df) == 0)

    with sel_col2:
        if is_active_empty:
            status_label = "⚠️ Empty (0 rows × 0 cols)"
            st.error(f"**Active Modeling Data:** `{source_name}` — **{status_label}** (ID: `{active_dataset_id[:24]}...`)")
        else:
            status_label = f"✨ Transformed ({len(active_df):,} rows × {active_df.shape[1]} cols)" if (is_working_selected and is_transformed) else f"📄 Raw Source ({len(active_df):,} rows × {active_df.shape[1]} cols)"
            st.info(f"**Active Modeling Data:** `{source_name}` — **{status_label}** (ID: `{active_dataset_id[:24]}...`)")

    # Stale experiment detection across dataset identity changes (Defect A2)
    last_trained_id = st.session_state.get("model_lab_active_dataset_id")
    dataset_has_changed = (last_trained_id is not None and last_trained_id != active_dataset_id)
    if dataset_has_changed:
        st.warning(f"⚠️ **Dataset State Changed:** The active modeling dataset has changed (previous: `{str(last_trained_id)[:20]}...`, active: `{active_dataset_id[:20]}...`). Prior experiment results are preserved below but marked as historical.")

    if is_active_empty:
        st.warning("⚠️ **Active Working Dataset is Empty:** A prior cleaning or filtering operation has removed all rows. Downstream modeling and recommendation actions are disabled until data is restored in **Clean & Transform**.")

    st.markdown("---")

    # Primary Modes Subtabs (Consolidating Quick Netflix Recommendations - Defect B1)
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "🎯 Mode 1: Predictive Modeling Studio",
        "🔍 Mode 2: Recommender Experiment Arena",
        "⚡ Mode 3: Quick Netflix Recommendations",
        "📋 Experiment Tracker & Export Hub",
    ])

    # ==================================================================
    # MODE 1: PREDICTIVE MODELING STUDIO
    # ==================================================================
    with ml_tab1:
        st.markdown("### 🎯 Predictive Modeling Studio")
        st.caption("Train supervised classification and regression pipelines on structured tabular features. Evaluates estimators on identical held-out test splits without data leakage.")

        if is_active_empty:
            st.info("Please load or restore rows in your dataset to configure and train predictive models.")
        else:
            p_cfg_col, p_res_col = st.columns([1, 2])

            with p_cfg_col:
                st.markdown("##### ⚙️ Task & Features")

                # Target Column Selection
                all_cols = list(active_df.columns)
                target_col = st.selectbox(
                    "Select Target Column:",
                    options=all_cols,
                    index=len(all_cols) - 1,
                    key="sb_target_col"
                )

                # Task Selection
                suggested_task = detect_suggested_task(active_df[target_col])
                task_type = st.radio(
                    "Supervised Task Type:",
                    options=["classification", "regression"],
                    index=0 if suggested_task == "classification" else 1,
                    horizontal=True,
                    help=f"Auto-suggested: {suggested_task.capitalize()} based on column type and distinct value count."
                )

                # Feature Columns Multiselect
                default_features = [c for c in all_cols if c != target_col][:8]
                selected_features = st.multiselect(
                    "Input Feature Columns:",
                    options=[c for c in all_cols if c != target_col],
                    default=default_features,
                    key="ms_features"
                )

                # Train/Test Split & Random Seed
                split_ratio = st.slider("Test Split Ratio:", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
                seed_val = st.number_input("Random Seed:", value=42, step=1)
                cv_folds = st.selectbox("Cross-Validation Folds (on train partition):", options=[0, 3, 5], index=1)

                st.markdown("##### 🤖 Candidate Models")
                if task_type == "classification":
                    run_baseline = st.checkbox("Baseline (Dummy Majority Class)", value=True)
                    run_linear = st.checkbox("Logistic Regression", value=True)
                    run_tree = st.checkbox("Decision Tree Classifier", value=True)
                    run_rf = st.checkbox("Random Forest Classifier", value=True)
                else:
                    run_baseline = st.checkbox("Baseline (Dummy Mean Regressor)", value=True)
                    run_linear = st.checkbox("Ridge Linear Regression", value=True)
                    run_tree = st.checkbox("Decision Tree Regressor", value=True)
                    run_rf = st.checkbox("Random Forest Regressor", value=True)

                tree_max_depth = st.slider("Tree / Forest Max Depth:", min_value=2, max_value=15, value=5)
                rf_n_estimators = st.slider("Random Forest Trees:", min_value=20, max_value=150, value=50, step=10)

                train_btn = st.button("🚀 Train & Compare Models", type="primary", use_container_width=True, disabled=is_active_empty)

            with p_res_col:
                if train_btn:
                    if not selected_features:
                        st.error("Please select at least one feature column.")
                    else:
                        try:
                            with st.spinner("Validating dataset and training modeling pipelines..."):
                                modeling_df, meta = validate_modeling_inputs(
                                    active_df,
                                    target_col=target_col,
                                    feature_cols=selected_features,
                                    task_type=task_type
                                )

                                for warn_msg in meta.get("warnings", []):
                                    st.warning(f"⚠️ {warn_msg}")

                                if meta["excluded_text_features"]:
                                    st.info(f"ℹ️ **Text Columns Excluded:** `{', '.join(meta['excluded_text_features'])}` were excluded from tabular modeling. Use **Mode 2: Recommender Arena** for NLP metadata.")

                                if meta["excluded_target_missing"] > 0:
                                    st.warning(f"⚠️ Excluded **{meta['excluded_target_missing']:,}** records with missing or invalid target values.")

                                # Split data with rare-class safeguard (Defect A8)
                                X = modeling_df[meta["usable_features"]]
                                y = modeling_df[target_col]

                                stratify_y = None
                                if task_type == "classification":
                                    class_counts = y.value_counts()
                                    if (class_counts >= 2).all() and len(y) * split_ratio >= len(class_counts):
                                        stratify_y = y
                                    else:
                                        st.caption("ℹ️ Stratification omitted: class frequencies or test split size are incompatible with stratified sampling.")

                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=split_ratio, random_state=int(seed_val), stratify=stratify_y
                                )

                                raw_test_df = modeling_df.loc[X_test.index]

                                # Train selected models
                                results = []
                                models_to_run = []
                                if run_baseline:
                                    models_to_run.append("baseline")
                                if run_linear:
                                    models_to_run.append("logistic" if task_type == "classification" else "linear")
                                if run_tree:
                                    models_to_run.append("decision_tree")
                                if run_rf:
                                    models_to_run.append("random_forest")

                                for m_name in models_to_run:
                                    res = train_and_evaluate_model(
                                        model_name=m_name,
                                        task_type=task_type,
                                        X_train=X_train,
                                        y_train=y_train,
                                        X_test=X_test,
                                        y_test=y_test,
                                        numeric_cols=meta["numeric_features"],
                                        categorical_cols=meta["categorical_features"],
                                        random_state=int(seed_val),
                                        cv_folds=cv_folds,
                                        dataset_identity=active_dataset_id,
                                        max_depth=tree_max_depth,
                                        n_estimators=rf_n_estimators
                                    )
                                    results.append(res)

                                # Save active run state
                                current_fp = compute_model_config_fingerprint(
                                    dataset_identity=active_dataset_id,
                                    target_col=target_col,
                                    task_type=task_type,
                                    feature_cols=selected_features,
                                    test_size=split_ratio,
                                    random_state=int(seed_val),
                                    cv_folds=cv_folds,
                                    model_name=models_to_run[0] if models_to_run else "baseline",
                                    hyperparameters={"max_depth": tree_max_depth, "n_estimators": rf_n_estimators},
                                    preprocessing_config={"max_categories": 50, "sparse": True},
                                    modeling_rows=len(X_train) + len(X_test)
                                )

                                st.session_state["model_lab_last_results"] = results
                                st.session_state["model_lab_active_dataset_id"] = active_dataset_id
                                st.session_state["model_lab_last_fingerprint"] = current_fp
                                st.session_state["model_lab_last_meta"] = meta
                                st.session_state["model_lab_test_data"] = (X_test, y_test, raw_test_df)

                        except Exception as e:
                            st.error(f"Modeling error: {str(e)}")

                # Display Results if available
                stored_results = st.session_state.get("model_lab_last_results")
                if stored_results:
                    # Check configuration fingerprint freshness (Defect A3)
                    active_fp = compute_model_config_fingerprint(
                        dataset_identity=active_dataset_id,
                        target_col=target_col,
                        task_type=task_type,
                        feature_cols=selected_features,
                        test_size=split_ratio,
                        random_state=int(seed_val),
                        cv_folds=cv_folds,
                        model_name=stored_results[0]["model_name"],
                        hyperparameters={"max_depth": tree_max_depth, "n_estimators": rf_n_estimators},
                        preprocessing_config={"max_categories": 50, "sparse": True},
                        modeling_rows=len(active_df)
                    )
                    last_saved_fp = st.session_state.get("model_lab_last_fingerprint")
                    is_config_stale = (last_saved_fp is not None and last_saved_fp != active_fp)

                    if dataset_has_changed:
                        st.warning("⚠️ **Historical Results (Dataset Changed):** These results were produced on a prior version of the dataset. Run a new training cycle to update metrics.")
                    elif is_config_stale:
                        st.warning("🕒 **Historical Results (Configuration Changed):** Current target, features, split, or hyperparameter selections have changed since this run. Click 'Train & Compare Models' to compute metrics for your newly selected settings.")

                    st.markdown("#### 📊 Model Evaluation & Comparison Table")
                    comp_rows = []
                    for r in stored_results:
                        row = {
                            "Model": r["model_name"].replace("_", " ").title(),
                            "Fit Time (s)": r["fit_duration_sec"],
                        }
                        row.update(r["metrics"])
                        comp_rows.append(row)

                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    # Visual Metric Bar Chart
                    metric_key = "macro_f1" if stored_results[0]["task_type"] == "classification" else "r2"
                    fig_comp = px.bar(
                        comp_df,
                        x="Model",
                        y=metric_key,
                        color="Model",
                        title=f"Candidate Model Comparison ({metric_key.upper()})",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_comp.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # Drill-Down Analysis for Top Performing Model
                    st.markdown("---")
                    st.markdown("#### 🔬 Detailed Inspection & Error Analysis")
                    model_options = [r["model_name"] for r in stored_results]
                    selected_model_name = st.selectbox("Inspect Fitted Model:", options=model_options)
                    chosen_res = next((r for r in stored_results if r["model_name"] == selected_model_name), stored_results[0])

                    diag_c1, diag_c2 = st.columns(2)
                    with diag_c1:
                        if chosen_res["task_type"] == "classification":
                            st.markdown("**Confusion Matrix:**")
                            cm = np.array(chosen_res["extra"]["confusion_matrix"])
                            lbls = chosen_res["extra"]["labels"]
                            fig_cm = px.imshow(
                                cm,
                                x=lbls,
                                y=lbls,
                                color_continuous_scale="Blues",
                                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                text_auto=True
                            )
                            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_cm, use_container_width=True)
                        else:
                            st.markdown("**Residuals Distribution:**")
                            res_vals = chosen_res["extra"]["residuals"]
                            fig_res = px.histogram(
                                res_vals,
                                nbins=25,
                                title="Prediction Errors (Actual - Predicted)",
                                color_discrete_sequence=["#2C7BE5"]
                            )
                            fig_res.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig_res, use_container_width=True)

                    with diag_c2:
                        st.markdown("**Feature Importances / Coefficients:**")
                        feat_imp = chosen_res["feature_importances"]
                        if feat_imp:
                            imp_df = pd.DataFrame(feat_imp).head(10)
                            fig_imp = px.bar(
                                imp_df,
                                x="importance",
                                y="feature",
                                orientation="h",
                                title="Top Influential Features",
                                color_discrete_sequence=["#00D97E"]
                            )
                            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig_imp, use_container_width=True)
                            st.caption("ℹ️ *Note: Feature importance represents statistical association within the fitted estimator, not proof of causation.*")
                        else:
                            st.info("Feature importance not applicable for this baseline estimator.")

                    # Error Analysis Table
                    test_tup = st.session_state.get("model_lab_test_data")
                    if test_tup:
                        X_test_ref, y_test_ref, raw_test_ref = test_tup
                        st.markdown("##### 🔎 Top Model Errors (Test Set Inspection)")
                        err_df = compute_error_analysis(
                            chosen_res["fitted_pipeline"],
                            X_test_ref,
                            y_test_ref,
                            raw_test_ref,
                            task_type=chosen_res["task_type"],
                            top_n=5
                        )
                        st.dataframe(err_df, use_container_width=True)

                    # Save Experiment Button with deduplication (Defect A7)
                    if st.button("💾 Save Experiment Run to History", key="btn_save_model_exp"):
                        saved_list = st.session_state.setdefault("model_lab_saved_experiments", [])
                        new_id = f"EXP_{len(saved_list)+1:03d}"
                        saved_entry = {
                            "experiment_id": new_id,
                            "dataset_id": active_dataset_id,
                            "dataset_name": source_name,
                            "source_mode": "working" if is_working_selected else "raw",
                            "task_type": chosen_res["task_type"],
                            "target_col": target_col,
                            "selected_features": selected_features,
                            "model": chosen_res["model_name"],
                            "model_name": chosen_res["model_name"],
                            "metrics": chosen_res["metrics"],
                            "fit_duration_sec": chosen_res["fit_duration_sec"],
                            "predict_duration_sec": chosen_res["predict_duration_sec"],
                            "test_size": split_ratio,
                            "random_state": int(seed_val),
                            "cv_folds": cv_folds,
                            "config_fingerprint": chosen_res.get("config_fingerprint", ""),
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        # Avoid duplicates on reruns
                        if not any(e.get("config_fingerprint") == saved_entry["config_fingerprint"] and e.get("model") == saved_entry["model"] for e in saved_list):
                            saved_list.append(saved_entry)
                            st.success(f"Experiment saved as `{saved_entry['experiment_id']}`!")
                        else:
                            st.info("This exact experiment run has already been recorded in history.")

    # ==================================================================
    # MODE 2: RECOMMENDER EXPERIMENT ARENA
    # ==================================================================
    with ml_tab2:
        st.markdown("### 🔍 Recommender Experiment Arena")
        st.caption("Investigate how altering NLP text representations, field weights, and vocabulary thresholds affects content-based recommendations. Employs on-demand sparse similarity without memory-intensive full matrix allocations.")

        if is_active_empty:
            st.info("Recommender experiments require an active non-empty dataset.")
        else:
            r_cfg_col, r_res_col = st.columns([1, 2])

            with r_cfg_col:
                st.markdown("##### ⚙️ Representation Settings")

                all_cols = list(active_df.columns)
                title_col = st.selectbox(
                    "Title / Item Column:",
                    options=all_cols,
                    index=all_cols.index("title") if "title" in all_cols else 0,
                    key="sb_rec_title_col"
                )

                potential_text_cols = [
                    c for c in all_cols
                    if c != title_col and (pd.api.types.is_object_dtype(active_df[c]) or pd.api.types.is_string_dtype(active_df[c]))
                ]
                default_text_fields = [c for c in ["listed_in", "director", "cast", "country", "description"] if c in potential_text_cols]
                if not default_text_fields:
                    default_text_fields = potential_text_cols[:3]

                selected_text_fields = st.multiselect(
                    "Metadata Text Fields:",
                    options=potential_text_cols,
                    default=default_text_fields,
                    key="ms_rec_text_fields"
                )

                st.markdown("##### ⚖️ Field Weights & Ablation")
                field_weights = {}
                for f in selected_text_fields:
                    field_weights[f] = st.slider(f"Weight for '{f}':", min_value=0.0, max_value=3.0, value=1.0, step=0.5, key=f"sl_w_{f}")

                max_feat = st.select_slider("TF-IDF Max Features (per field):", options=[500, 1000, 3000, 5000, 10000], value=3000)
                use_stopwords = st.checkbox("English Stop Words", value=True)
                use_bigrams = st.checkbox("Include Bigrams (1, 2)", value=False)
                top_k_val = st.slider("Top-K Recommendations:", min_value=3, max_value=20, value=5)

                st.markdown("##### 🎯 Query Selection")
                catalog_titles = active_df[title_col].dropna().astype(str).tolist()
                default_query_idx = 0
                if "Inception" in catalog_titles:
                    default_query_idx = catalog_titles.index("Inception")
                elif "Stranger Things" in catalog_titles:
                    default_query_idx = catalog_titles.index("Stranger Things")

                query_title = st.selectbox("Select Query Title:", options=catalog_titles, index=default_query_idx, key="sb_query_title")

                run_rec_btn = st.button("⚡ Run Recommendation Experiment", type="primary", use_container_width=True, disabled=is_active_empty)

            with r_res_col:
                if run_rec_btn:
                    if not selected_text_fields:
                        st.error("Please select at least one metadata text field.")
                    else:
                        try:
                            with st.spinner("Fitting sparse TF-IDF vectors and computing query similarities..."):
                                recommender = RecommenderArenaService(
                                    df=active_df,
                                    title_col=title_col,
                                    text_fields=selected_text_fields,
                                    field_weights=field_weights,
                                    max_features=max_feat,
                                    stop_words="english" if use_stopwords else None,
                                    ngram_range=(1, 2) if use_bigrams else (1, 1)
                                )
                                rec_results, query_time = recommender.recommend(query_title, top_k=top_k_val)

                                current_run = {
                                    "query_title": query_title,
                                    "title_col": title_col,
                                    "rec_results": rec_results,
                                    "text_fields": selected_text_fields,
                                    "field_weights": field_weights,
                                    "max_features": max_feat,
                                    "vocab_size": recommender.vocabulary_size,
                                    "nnz_entries": recommender.nnz_entries,
                                    "fit_time": recommender.fit_duration_sec,
                                    "query_time": query_time,
                                    "top_k": top_k_val,
                                    "dataset_id": active_dataset_id,
                                }

                                if "rec_current_run" in st.session_state:
                                    st.session_state["rec_previous_run"] = st.session_state["rec_current_run"]
                                st.session_state["rec_current_run"] = current_run

                        except Exception as e:
                            st.error(f"Recommender error: {str(e)}")

                active_rec_run = st.session_state.get("rec_current_run")
                if active_rec_run:
                    # Check dataset consistency for recommender run
                    if active_rec_run.get("dataset_id") != active_dataset_id:
                        st.warning("⚠️ **Historical Recommender Run:** Current recommendations were computed on an earlier dataset state. Re-run experiment to update.")

                    st.markdown(f"#### 🎬 Top Recommendations for: *{active_rec_run['query_title']}*")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Vocabulary Dimension", f"{active_rec_run['vocab_size']:,} terms")
                    q2.metric("Sparse Non-Zero Elements", f"{active_rec_run['nnz_entries']:,}")
                    q3.metric("Query Latency", f"{active_rec_run['query_time']*1000:.1f} ms")

                    res_table = active_rec_run["rec_results"]
                    if res_table.empty:
                        st.info("No recommendations found matching the query item.")
                    else:
                        display_cols = ["rank", title_col, "similarity_score"] + [c for c in ["listed_in", "director", "release_year"] if c in res_table.columns]
                        st.dataframe(res_table[display_cols], use_container_width=True, hide_index=True)
                        st.caption("ℹ️ *Cosine similarity scores represent vector angles in term space, not predicted user ratings or recommendation probabilities.*")

                        # A/B Comparison against previous configuration with verification (Defect A6)
                        prev_run = st.session_state.get("rec_previous_run")
                        if prev_run:
                            same_query = prev_run.get("query_title") == active_rec_run["query_title"]
                            same_ds = prev_run.get("dataset_id") == active_rec_run["dataset_id"]
                            same_col = prev_run.get("title_col") == active_rec_run["title_col"]

                            if same_query and same_ds and same_col:
                                st.markdown("---")
                                st.markdown("#### ⚖️ A/B Comparison vs. Previous Run")
                                ab_diff = compare_recommender_runs(prev_run["rec_results"], active_rec_run["rec_results"], title_col=title_col)

                                m1, m2, m3 = st.columns(3)
                                m1.metric("Catalog Jaccard Overlap", f"{ab_diff['overlap_pct']}%")
                                m2.metric("Shared Top-K Titles", f"{ab_diff['shared_count']}")
                                m3.metric("Entering Top-K", f"{len(ab_diff['entering_items'])} titles")

                                if ab_diff["entering_items"]:
                                    st.info(f"✨ **Items Entering Top-K:** `{', '.join(ab_diff['entering_items'])}`")
                                if ab_diff["leaving_items"]:
                                    st.warning(f"🗑️ **Items Leaving Top-K:** `{', '.join(ab_diff['leaving_items'])}`")
                            elif not same_ds:
                                st.caption("ℹ️ A/B comparison skipped: baseline run was evaluated on a different dataset version.")

                        # Truthful User Relevance Feedback Evaluation (Defect A6)
                        st.markdown("---")
                        st.markdown("#### 🎯 User Relevance Feedback Evaluation")
                        st.caption("Benchmark recommendations without fabricated ground truth. Mark items as relevant or irrelevant to compute judged relevance rate and partial NDCG.")

                        # Key feedback store by (dataset_id, title_col, query_title) to prevent cross-dataset contamination
                        feedback_key = f"{active_rec_run['dataset_id']}::{title_col}::{active_rec_run['query_title']}"
                        all_feedback_store = st.session_state.setdefault("recommender_feedback_master", {})
                        query_feedback = all_feedback_store.setdefault(feedback_key, {})

                        rec_titles = res_table[title_col].tolist()
                        for t in rec_titles:
                            fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([3, 1, 1, 1])
                            with fb_col1:
                                current_status = query_feedback.get(t)
                                status_emoji = "👍 Relevant" if current_status == 1 else ("👎 Irrelevant" if current_status == 0 else "⚪ Unjudged")
                                st.markdown(f"**{t}** — `{status_emoji}`")
                            with fb_col2:
                                if st.button("👍", key=f"fb_pos_{feedback_key}_{t}"):
                                    query_feedback[t] = 1
                                    st.rerun()
                            with fb_col3:
                                if st.button("👎", key=f"fb_neg_{feedback_key}_{t}"):
                                    query_feedback[t] = 0
                                    st.rerun()
                            with fb_col4:
                                if st.button("Clear", key=f"fb_clr_{feedback_key}_{t}"):
                                    query_feedback.pop(t, None)
                                    st.rerun()

                        eval_metrics = compute_relevance_feedback_metrics(rec_titles, query_feedback)
                        ev1, ev2, ev3 = st.columns(3)
                        ev1.metric(
                            "Relevance Rate Among Judged",
                            f"{eval_metrics['relevance_rate_among_judged']}%" if eval_metrics['relevance_rate_among_judged'] is not None else "Pending",
                            help=eval_metrics.get("relevance_rate_denominator", "")
                        )
                        ev2.metric(
                            "Partial NDCG@K",
                            f"{eval_metrics['ndcg_at_k']}" if eval_metrics['ndcg_at_k'] is not None else "Pending",
                            help=eval_metrics.get("ndcg_method", "")
                        )
                        ev3.metric("Evaluated Items", f"{eval_metrics['judged_count']} / {eval_metrics['total_returned']}")
                        st.caption(f"ℹ️ {eval_metrics['disclosure']}")

                        fb_btn_c1, fb_btn_c2 = st.columns(2)
                        with fb_btn_c1:
                            if st.button("🧹 Clear Feedback for Current Query", key="btn_clr_query_fb"):
                                query_feedback.clear()
                                st.rerun()
                        with fb_btn_c2:
                            if st.button("🗑️ Reset All Evaluation State", key="btn_reset_all_fb"):
                                all_feedback_store.clear()
                                st.rerun()

                        # Save Recommender Run
                        if st.button("💾 Save Recommender Run to History", key="btn_save_rec_run"):
                            saved_recs = st.session_state.setdefault("model_lab_saved_recommenders", [])
                            new_rec_id = f"REC_{len(saved_recs)+1:03d}"
                            saved_rec = {
                                "run_id": new_rec_id,
                                "experiment_id": new_rec_id,
                                "query_title": active_rec_run["query_title"],
                                "title_col": title_col,
                                "dataset_id": active_dataset_id,
                                "fields": active_rec_run["text_fields"],
                                "weights": active_rec_run["field_weights"],
                                "vocab_size": active_rec_run["vocab_size"],
                                "top_titles": rec_titles,
                                "top_k": active_rec_run.get("top_k", 5),
                                "fit_duration_sec": active_rec_run["fit_time"],
                                "query_duration_sec": active_rec_run["query_time"],
                                "relevance_judgments": dict(query_feedback),
                                "relevance_metrics": eval_metrics,
                                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            saved_recs.append(saved_rec)
                            st.success(f"Saved recommender run as `{saved_rec['run_id']}`!")

    # ==================================================================
    # MODE 3: QUICK NETFLIX RECOMMENDATIONS (CONSOLIDATED - DEFECT B1)
    # ==================================================================
    with ml_tab3:
        st.markdown("### ⚡ Quick Netflix Recommendations")
        st.caption("Consolidated catalog recommender mode using established Netflix metadata soup features with executive card presentation.")

        # Check if dataset has title and description/genres
        has_title = "title" in active_df.columns
        if not has_title:
            st.warning("⚠️ Quick Netflix Recommendations requires a 'title' column in the dataset. Switch to Mode 2 for custom columns.")
        else:
            avail_titles = sorted(active_df["title"].dropna().unique().tolist())
            def_idx = 0
            for pref in ["Stranger Things", "Inception", "Breaking Bad", "Shadows over Silence"]:
                if pref in avail_titles:
                    def_idx = avail_titles.index(pref)
                    break

            quick_sel_col, quick_info_col = st.columns([3, 1])
            with quick_sel_col:
                selected_q_title = st.selectbox(
                    "Select Seed Title for Instant Recommendations:",
                    avail_titles,
                    index=def_idx,
                    key="quick_rec_seed_title",
                )
            with quick_info_col:
                st.markdown(f"""
                    <div style="background: rgba(20, 28, 46, 0.85); border: 1px solid #1E2B44; border-radius: 8px; padding: 10px 14px; margin-top: 24px;">
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Active Catalog Corpus</div>
                        <div style="font-size: 1.15rem; color: #F8FAFC; font-weight: 700;">{len(active_df):,} <span style="font-size: 0.8rem; font-weight: 500; color: #00D2FF;">Titles</span></div>
                    </div>
                """, unsafe_allow_html=True)

            if selected_q_title:
                target_rows = active_df[active_df["title"] == selected_q_title]
                if not target_rows.empty:
                    target_r = target_rows.iloc[0]
                    t_year = int(target_r["release_year"]) if pd.notna(target_r.get("release_year")) else "N/A"
                    t_rating = str(target_r.get("rating", "NR"))
                    t_type = str(target_r.get("type", "Title"))
                    t_genres = str(target_r.get("listed_in", "Uncategorized"))
                    t_director = str(target_r.get("director", "Unlisted"))
                    t_cast = str(target_r.get("cast", "Unlisted"))
                    t_desc = str(target_r.get("description", "No description available."))

                    st.markdown(f"""
                        <div style="background: linear-gradient(180deg, rgba(20, 28, 46, 0.95) 0%, rgba(14, 20, 34, 0.95) 100%); border: 1px solid #1E2B44; border-left: 4px solid #00D2FF; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
                            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                    <span style="background: rgba(0, 210, 255, 0.15); color: #00D2FF; border: 1px solid rgba(0, 210, 255, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">QUERY TITLE</span>
                                    <span style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">{target_r['title']}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="background: rgba(148, 163, 184, 0.12); color: #CBD5E1; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">{t_type}</span>
                                    <span style="background: rgba(148, 163, 184, 0.12); color: #CBD5E1; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">{t_rating}</span>
                                    <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">{t_year}</span>
                                </div>
                            </div>
                            <div style="color: #94A3B8; font-size: 0.85rem; margin: 8px 0;">
                                <b>Genres:</b> {t_genres} &nbsp;|&nbsp; <b>Director:</b> {t_director} &nbsp;|&nbsp; <b>Cast:</b> {t_cast}
                            </div>
                            <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.5;">{t_desc}</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Fit memory-safe ContentRecommender
                @st.cache_resource(show_spinner="Fitting Quick Content Recommender...")
                def _cached_recommender(data_to_fit: pd.DataFrame) -> ContentRecommender:
                    rec = ContentRecommender(max_features=5000)
                    rec.fit(data_to_fit)
                    return rec

                quick_model = _cached_recommender(active_df)
                quick_recs = quick_model.recommend(selected_q_title, top_n=5)

                st.markdown(f"#### Recommended Titles for *'{selected_q_title}'*")
                if quick_recs.empty:
                    st.info("No similar titles found within the current dataset.")
                else:
                    for rank, (_, row) in enumerate(quick_recs.iterrows(), 1):
                        sim_pct = round(row.get("similarity_score", row.get("similarity", 0.0)) * 100, 1)
                        r_rating = str(row.get("rating", "NR"))
                        r_type = str(row.get("type", "Title"))
                        r_year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A"
                        r_desc = str(row.get("description", "No description available."))
                        r_genres = str(row.get("listed_in", "Uncategorized"))
                        r_director = str(row.get("director", "Unlisted"))
                        r_cast = str(row.get("cast", "Unlisted"))

                        st.markdown(f"""
                            <div style="background: linear-gradient(180deg, rgba(20, 28, 46, 0.95) 0%, rgba(14, 20, 34, 0.95) 100%); border: 1px solid #1E2B44; border-left: 4px solid #00D2FF; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <span style="background: rgba(0, 210, 255, 0.15); color: #00D2FF; border: 1px solid rgba(0, 210, 255, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">MATCH #{rank}</span>
                                        <span style="font-size: 1.12rem; font-weight: 700; color: #F8FAFC;">{row['title']}</span>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span style="background: rgba(0, 210, 255, 0.12); color: #00D2FF; border: 1px solid rgba(0, 210, 255, 0.25); padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 700;">{sim_pct}% MATCH</span>
                                        <span style="background: rgba(148, 163, 184, 0.12); color: #CBD5E1; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 600;">{r_type}</span>
                                        <span style="background: rgba(148, 163, 184, 0.12); color: #CBD5E1; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 600;">{r_rating}</span>
                                        <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">{r_year}</span>
                                    </div>
                                </div>
                                <div style="color: #94A3B8; font-size: 0.83rem; margin: 6px 0;">
                                    <b>Genres:</b> {r_genres} &nbsp;|&nbsp; <b>Director:</b> {r_director} &nbsp;|&nbsp; <b>Cast:</b> {r_cast}
                                </div>
                                <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">{r_desc}</div>
                            </div>
                        """, unsafe_allow_html=True)

    # ==================================================================
    # SUBTAB 4: EXPERIMENT TRACKER & EXPORT HUB (DEFECT A7)
    # ==================================================================
    with ml_tab4:
        st.markdown("### 📋 Experiment Tracker & Export Hub")
        st.caption("Review, compare, delete, and export all predictive modeling and recommendation runs saved within this session.")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("##### 🎯 Predictive Modeling History")
            saved_models = st.session_state.get("model_lab_saved_experiments", [])
            if not saved_models:
                st.info("No predictive modeling experiments saved yet.")
            else:
                table_rows = []
                for exp in saved_models:
                    row = {
                        "ID": exp["experiment_id"],
                        "Model": exp["model"].title(),
                        "Task": exp["task_type"],
                        "Target": exp["target_col"],
                        "Time": exp["timestamp"],
                    }
                    row.update(exp.get("metrics", {}))
                    table_rows.append(row)
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                c_del1, c_del2 = st.columns([2, 1])
                with c_del1:
                    exp_to_del = st.selectbox("Select Experiment to Delete:", options=[e["experiment_id"] for e in saved_models], key="sb_del_model_exp")
                with c_del2:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete Selected", key="btn_del_model_exp"):
                        st.session_state["model_lab_saved_experiments"] = [e for e in saved_models if e["experiment_id"] != exp_to_del]
                        st.success(f"Removed `{exp_to_del}`.")
                        st.rerun()

                # Export CSV & JSON
                csv_buf = io.StringIO()
                pd.DataFrame(table_rows).to_csv(csv_buf, index=False)
                c_exp1, c_exp2 = st.columns(2)
                with c_exp1:
                    st.download_button(
                        "📥 Export Models (CSV)",
                        data=csv_buf.getvalue(),
                        file_name="predictive_model_experiments.csv",
                        mime="text/csv"
                    )
                with c_exp2:
                    json_str = json.dumps([json.loads(export_model_experiment_json(e)) for e in saved_models], indent=2)
                    st.download_button(
                        "📥 Export Models (JSON)",
                        data=json_str,
                        file_name="predictive_model_experiments.json",
                        mime="application/json"
                    )

        with exp_col2:
            st.markdown("##### 🔍 Recommender Experiments History")
            saved_recs = st.session_state.get("model_lab_saved_recommenders", [])
            if not saved_recs:
                st.info("No recommender experiment runs saved yet.")
            else:
                rec_rows = []
                for r in saved_recs:
                    rec_rows.append({
                        "ID": r["run_id"],
                        "Query Title": r["query_title"],
                        "Vocab Size": r["vocab_size"],
                        "Top Titles": ", ".join(r["top_titles"][:3]),
                        "Time": r["timestamp"],
                    })
                st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

                c_rdel1, c_rdel2 = st.columns([2, 1])
                with c_rdel1:
                    rec_to_del = st.selectbox("Select Recommender to Delete:", options=[r["run_id"] for r in saved_recs], key="sb_del_rec_exp")
                with c_rdel2:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete Selected", key="btn_del_rec_exp"):
                        st.session_state["model_lab_saved_recommenders"] = [r for r in saved_recs if r["run_id"] != rec_to_del]
                        st.success(f"Removed `{rec_to_del}`.")
                        st.rerun()

                csv_rec = io.StringIO()
                pd.DataFrame(rec_rows).to_csv(csv_rec, index=False)
                cr_exp1, cr_exp2 = st.columns(2)
                with cr_exp1:
                    st.download_button(
                        "📥 Export Recommenders (CSV)",
                        data=csv_rec.getvalue(),
                        file_name="recommender_experiments.csv",
                        mime="text/csv"
                    )
                with cr_exp2:
                    json_rec = json.dumps([json.loads(export_recommender_experiment_json(r)) for r in saved_recs], indent=2)
                    st.download_button(
                        "📥 Export Recommenders (JSON)",
                        data=json_rec,
                        file_name="recommender_experiments.json",
                        mime="application/json"
                    )
