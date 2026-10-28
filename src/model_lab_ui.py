"""
model_lab_ui.py
Streamlit UI presentation module for Phase 3 DataLab — Model Lab: ML Experiment Arena.

Features:
- Seamless shared dataset selection: Raw Source vs. Phase 2 Transformed Working Copy.
- Mode 1 — Predictive Modeling Studio: Supervised Classification & Regression with
  leakage-free preprocessing, baseline comparison, interactive Plotly confusion matrices /
  residual distributions, feature importance, and error analysis.
- Mode 2 — Recommender Experiment Arena: Memory-safe multi-field TF-IDF with field weighting,
  ablation, on-demand query similarity, A/B query comparisons, and user relevance feedback.
- Experiment Hub: Run logging, CSV export, and JSON configuration serialization.
"""

from typing import Dict, Any, List, Optional
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.cleaning_lab import (
    get_datalab_source_dataset,
    get_datalab_working_dataset,
)
from src.model_lab import (
    detect_suggested_task,
    validate_modeling_inputs,
    train_and_evaluate_model,
    compute_error_analysis,
    RecommenderArenaService,
    compare_recommender_runs,
    compute_relevance_feedback_metrics,
    export_model_experiment_json,
    export_recommender_experiment_json,
)

PRIMARY_COLOR = "#2C7BE5"
ACCENT_COLOR = "#6EA8FE"
SUCCESS_COLOR = "#00D97E"
DANGER_COLOR = "#E63757"
TEXT_COLOR = "#2B3035"


def render_model_lab_tab(netflix_raw_df: Optional[pd.DataFrame] = None) -> None:
    """
    Main entry point for the Model Lab: ML Experiment Arena tab.
    """
    st.markdown("""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h2 style="font-weight: 700; font-size: 1.85rem; color: #2B3035; margin-bottom: 2px;">
                        🤖 DataLab — Model Lab: ML Experiment Arena
                    </h2>
                    <p style="color: #6C757D; font-size: 0.95rem; margin-top: 0px;">
                        Train supervised predictive models on tabular features or experiment with NLP recommendation representations.
                        Compare models against baselines, inspect feature importances, and evaluate relevance without fabricated ground truth.
                    </p>
                </div>
                <span style="display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; background-color: #E7F1FF; color: #2C7BE5; border: 1px solid #B6D4FE;">
                    PHASE 3: MODEL LAB
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Step 1: Shared Dataset State Retrieval (Source vs Working Copy)
    # ------------------------------------------------------------------
    source_df, source_ident, source_name = get_datalab_source_dataset()
    working_df, working_ident, working_name, is_transformed = get_datalab_working_dataset()

    if source_df is None or source_df.empty:
        if netflix_raw_df is not None and not netflix_raw_df.empty:
            source_df = netflix_raw_df.copy()
            source_name = "Built-in Netflix Catalog"
            source_ident = "netflix_catalog"
            working_df = source_df.copy()
            working_ident = source_ident
            working_name = source_name
            is_transformed = False

    if source_df is None or source_df.empty:
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

    active_df = working_df if (data_choice == "Clean & Transform Working Copy" and working_df is not None) else source_df
    active_dataset_id = f"{working_ident}_working" if data_choice == "Clean & Transform Working Copy" else f"{source_ident}_source"

    with sel_col2:
        status_label = f"✨ Transformed ({len(working_df):,} rows × {working_df.shape[1]} cols)" if (data_choice == "Clean & Transform Working Copy" and is_transformed) else f"📄 Raw Source ({len(source_df):,} rows × {source_df.shape[1]} cols)"
        st.info(f"**Active Modeling Data:** `{source_name}` — **{status_label}** (ID: `{active_dataset_id[:16]}...`)")

    # Stale experiment detection
    last_trained_id = st.session_state.get("model_lab_active_dataset_id")
    if last_trained_id and last_trained_id != active_dataset_id:
        st.warning("⚠️ **Dataset Source Changed:** The underlying dataset identity has changed. Incompatible prior experiment models are marked as stale.")

    st.markdown("---")

    # Primary Modes Subtabs
    ml_tab1, ml_tab2, ml_tab3 = st.tabs([
        "🎯 Mode 1: Predictive Modeling Studio",
        "🔍 Mode 2: Recommender Experiment Arena",
        "📋 Experiment Tracker & Export Hub",
    ])

    # ==================================================================
    # MODE 1: PREDICTIVE MODELING STUDIO
    # ==================================================================
    with ml_tab1:
        st.markdown("### 🎯 Predictive Modeling Studio")
        st.caption("Train supervised classification and regression pipelines on structured tabular features. Evaluates estimators on identical held-out test splits without data leakage.")

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

            train_btn = st.button("🚀 Train & Compare Models", type="primary", use_container_width=True)

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

                            if meta["excluded_text_features"]:
                                st.info(f"ℹ️ **Text Columns Excluded:** `{', '.join(meta['excluded_text_features'])}` were excluded from tabular modeling. Use **Mode 2: Recommender Arena** for NLP metadata.")

                            if meta["excluded_target_missing"] > 0:
                                st.warning(f"⚠️ Excluded **{meta['excluded_target_missing']:,}** records with missing or invalid target values.")

                            # Split data
                            X = modeling_df[meta["usable_features"]]
                            y = modeling_df[target_col]

                            # Stratification check for classification
                            stratify_y = None
                            if task_type == "classification":
                                class_counts = y.value_counts()
                                if (class_counts >= 2).all():
                                    stratify_y = y
                                else:
                                    st.caption("ℹ️ Rare class (<2 examples) detected. Using standard random split without stratification.")

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
                                    max_depth=tree_max_depth,
                                    n_estimators=rf_n_estimators
                                )
                                results.append(res)

                            # Save to session state
                            st.session_state["model_lab_last_results"] = results
                            st.session_state["model_lab_active_dataset_id"] = active_dataset_id
                            st.session_state["model_lab_last_meta"] = meta
                            st.session_state["model_lab_test_data"] = (X_test, y_test, raw_test_df)

                    except Exception as e:
                        st.error(f"Modeling error: {str(e)}")

            # Display Results if available
            stored_results = st.session_state.get("model_lab_last_results")
            if stored_results:
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

                # Save Experiment Button
                if st.button("💾 Save Experiment Run to History", key="btn_save_model_exp"):
                    if "model_lab_saved_experiments" not in st.session_state:
                        st.session_state["model_lab_saved_experiments"] = []

                    saved_entry = {
                        "experiment_id": f"EXP_{len(st.session_state['model_lab_saved_experiments'])+1:03d}",
                        "dataset_id": active_dataset_id,
                        "dataset_name": source_name,
                        "task_type": chosen_res["task_type"],
                        "target_col": target_col,
                        "model": chosen_res["model_name"],
                        "metrics": chosen_res["metrics"],
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state["model_lab_saved_experiments"].append(saved_entry)
                    st.success(f"Experiment saved as `{saved_entry['experiment_id']}`!")

    # ==================================================================
    # MODE 2: RECOMMENDER EXPERIMENT ARENA
    # ==================================================================
    with ml_tab2:
        st.markdown("### 🔍 Recommender Experiment Arena")
        st.caption("Investigate how altering NLP text representations, field weights, and vocabulary thresholds affects content-based recommendations. Employs on-demand sparse similarity without memory-intensive full matrix allocations.")

        r_cfg_col, r_res_col = st.columns([1, 2])

        with r_cfg_col:
            st.markdown("##### ⚙️ Representation Settings")

            # Title / Item column
            all_cols = list(active_df.columns)
            title_col = st.selectbox(
                "Title / Item Column:",
                options=all_cols,
                index=all_cols.index("title") if "title" in all_cols else 0,
                key="sb_rec_title_col"
            )

            # Metadata Fields Multiselect
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

            # Field Weights
            st.markdown("##### ⚖️ Field Weights & Ablation")
            field_weights = {}
            for f in selected_text_fields:
                field_weights[f] = st.slider(f"Weight for '{f}':", min_value=0.0, max_value=3.0, value=1.0, step=0.5, key=f"sl_w_{f}")

            # TF-IDF Hyperparameters
            max_feat = st.select_slider("TF-IDF Max Features (per field):", options=[500, 1000, 3000, 5000, 10000], value=3000)
            use_stopwords = st.checkbox("English Stop Words", value=True)
            use_bigrams = st.checkbox("Include Bigrams (1, 2)", value=False)
            top_k_val = st.slider("Top-K Recommendations:", min_value=3, max_value=20, value=5)

            # Query Title
            st.markdown("##### 🎯 Query Selection")
            catalog_titles = active_df[title_col].dropna().astype(str).tolist()
            default_query_idx = 0
            if "Inception" in catalog_titles:
                default_query_idx = catalog_titles.index("Inception")

            query_title = st.selectbox("Select Query Title:", options=catalog_titles, index=default_query_idx, key="sb_query_title")

            run_rec_btn = st.button("⚡ Run Recommendation Experiment", type="primary", use_container_width=True)

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
                                "rec_results": rec_results,
                                "text_fields": selected_text_fields,
                                "field_weights": field_weights,
                                "max_features": max_feat,
                                "vocab_size": recommender.vocabulary_size,
                                "nnz_entries": recommender.nnz_entries,
                                "fit_time": recommender.fit_duration_sec,
                                "query_time": query_time,
                                "dataset_id": active_dataset_id,
                            }

                            # Save previous run as Baseline A if available
                            if "rec_current_run" in st.session_state:
                                st.session_state["rec_previous_run"] = st.session_state["rec_current_run"]
                            st.session_state["rec_current_run"] = current_run

                    except Exception as e:
                        st.error(f"Recommender error: {str(e)}")

            active_rec_run = st.session_state.get("rec_current_run")
            if active_rec_run:
                st.markdown(f"#### 🎬 Top Recommendations for: *{active_rec_run['query_title']}*")
                q1, q2, q3 = st.columns(3)
                q1.metric("Vocabulary Dimension", f"{active_rec_run['vocab_size']:,} terms")
                q2.metric("Sparse Non-Zero Elements", f"{active_rec_run['nnz_entries']:,}")
                q3.metric("Query Latency", f"{active_rec_run['query_time']*1000:.1f} ms")

                res_table = active_rec_run["rec_results"]
                display_cols = ["rank", title_col, "similarity_score"] + [c for c in ["listed_in", "director", "release_year"] if c in res_table.columns]
                st.dataframe(res_table[display_cols], use_container_width=True, hide_index=True)
                st.caption("ℹ️ *Cosine similarity scores represent vector angles in term space, not predicted user ratings or recommendation probabilities.*")

                # A/B Comparison against previous configuration
                prev_run = st.session_state.get("rec_previous_run")
                if prev_run and prev_run["query_title"] == active_rec_run["query_title"]:
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

                # User Relevance Feedback Evaluation
                st.markdown("---")
                st.markdown("#### 🎯 User Relevance Feedback Evaluation")
                st.caption("Benchmark recommendations without fabricated ground truth. Mark items as relevant or irrelevant to compute judged Precision@K and NDCG@K.")

                feedback_store = st.session_state.setdefault("rec_user_feedback", {})
                current_query = active_rec_run["query_title"]
                query_feedback = feedback_store.setdefault(current_query, {})

                rec_titles = res_table[title_col].tolist()
                for t in rec_titles:
                    fb_col1, fb_col2, fb_col3 = st.columns([3, 1, 1])
                    with fb_col1:
                        current_status = query_feedback.get(t)
                        status_emoji = "👍 Relevant" if current_status == 1 else ("👎 Irrelevant" if current_status == 0 else "⚪ Unjudged")
                        st.markdown(f"**{t}** — `{status_emoji}`")
                    with fb_col2:
                        if st.button("👍", key=f"fb_pos_{current_query}_{t}"):
                            query_feedback[t] = 1
                            st.rerun()
                    with fb_col3:
                        if st.button("👎", key=f"fb_neg_{current_query}_{t}"):
                            query_feedback[t] = 0
                            st.rerun()

                eval_metrics = compute_relevance_feedback_metrics(rec_titles, query_feedback)
                ev1, ev2, ev3 = st.columns(3)
                ev1.metric("Judged Precision@K", f"{eval_metrics['precision_at_k']}%" if eval_metrics['precision_at_k'] is not None else "Pending")
                ev2.metric("Judged NDCG@K", f"{eval_metrics['ndcg_at_k']}" if eval_metrics['ndcg_at_k'] is not None else "Pending")
                ev3.metric("Evaluated Items", f"{eval_metrics['judged_count']} / {eval_metrics['total_returned']}")
                st.caption(f"ℹ️ {eval_metrics['disclosure']}")

                # Save Recommender Run
                if st.button("💾 Save Recommender Run to History", key="btn_save_rec_run"):
                    if "model_lab_saved_recommenders" not in st.session_state:
                        st.session_state["model_lab_saved_recommenders"] = []
                    saved_rec = {
                        "run_id": f"REC_{len(st.session_state['model_lab_saved_recommenders'])+1:03d}",
                        "query_title": active_rec_run["query_title"],
                        "dataset_id": active_dataset_id,
                        "fields": active_rec_run["text_fields"],
                        "weights": active_rec_run["field_weights"],
                        "vocab_size": active_rec_run["vocab_size"],
                        "top_titles": rec_titles,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state["model_lab_saved_recommenders"].append(saved_rec)
                    st.success(f"Saved recommender run as `{saved_rec['run_id']}`!")

    # ==================================================================
    # SUBTAB 3: EXPERIMENT TRACKER & EXPORT HUB
    # ==================================================================
    with ml_tab3:
        st.markdown("### 📋 Experiment Tracker & Export Hub")
        st.caption("Review and export all predictive modeling and recommendation runs saved within this session.")

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
                    row.update(exp["metrics"])
                    table_rows.append(row)
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                # Export CSV
                csv_buf = io.StringIO()
                pd.DataFrame(table_rows).to_csv(csv_buf, index=False)
                st.download_button(
                    "📥 Export Model Experiments (CSV)",
                    data=csv_buf.getvalue(),
                    file_name="predictive_model_experiments.csv",
                    mime="text/csv"
                )

                # Export JSON
                json_str = json.dumps(saved_models, indent=2, default=str)
                st.download_button(
                    "📥 Export Model Experiments (JSON)",
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

                # Export CSV
                csv_rec = io.StringIO()
                pd.DataFrame(rec_rows).to_csv(csv_rec, index=False)
                st.download_button(
                    "📥 Export Recommender Runs (CSV)",
                    data=csv_rec.getvalue(),
                    file_name="recommender_experiments.csv",
                    mime="text/csv"
                )

                # Export JSON
                json_rec = json.dumps(saved_recs, indent=2, default=str)
                st.download_button(
                    "📥 Export Recommender Runs (JSON)",
                    data=json_rec,
                    file_name="recommender_experiments.json",
                    mime="application/json"
                )
