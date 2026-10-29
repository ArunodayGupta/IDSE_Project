"""
missions.py
Phase 4: Data Scientist Mission Mode System & Mission Registry.

Defines:
- DataScientistMission class declaring objectives, hints, verification criteria, and reflections.
- Complete 7-mission curriculum grounded in verifiable user actions across DataLab.
- Dynamic dataset adaptation (Netflix-specific vs generic uploaded tabular data).
- Session-based progress tracking and evidence recording.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd


class DataScientistMission:
    """Represents a structured, verifiable data science guided mission."""

    def __init__(
        self,
        mission_id: str,
        title: str,
        difficulty: str,
        objective: str,
        steps: List[str],
        hints: Dict[int, str],
        verification_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[bool, str, Dict[str, Any]]],
        reflection_question: Dict[str, Any],
        tradeoff_explanation: str,
        badge: str,
        required_columns: Optional[List[str]] = None,
        next_mission_id: Optional[str] = None
    ):
        self.mission_id = mission_id
        self.title = title
        self.difficulty = difficulty
        self.objective = objective
        self.steps = steps
        self.hints = hints
        self.verification_fn = verification_fn
        self.reflection_question = reflection_question
        self.tradeoff_explanation = tradeoff_explanation
        self.badge = badge
        self.required_columns = required_columns or []
        self.next_mission_id = next_mission_id

    def is_compatible(self, df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
        """Checks whether the current dataset satisfies required prerequisites."""
        if df is None or len(df) == 0:
            return False, "Active dataset contains 0 rows. Please select or load a non-empty dataset."
        if self.required_columns:
            missing = [c for c in self.required_columns if c not in df.columns]
            if missing:
                return False, f"Mission requires columns: {', '.join(missing)} which are missing in this dataset schema."
        return True, ""


# =====================================================================
# VERIFICATION FUNCTIONS (GROUNDED IN REAL EVIDENCE - DEFECT D1)
# =====================================================================

def verify_mission_1(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 1: Verify dataset investigation (checked overview or missingness)."""
    explored = session_state.get("datalab_user_explored", False)
    # Also verify if user interacted with filter/explore state or if dataset is profiled
    if explored or session_state.get("explorer_active_filter_count", 0) > 0 or session_state.get("explorer_view_mode"):
        return True, "Verified: Dataset dimensions, schema, and column distributions were successfully inspected.", {
            "rows_inspected": len(df),
            "columns_profiled": len(df.columns),
            "evidence": "Exploration interaction confirmed."
        }
    return False, "Incomplete: Visit 'Explore Dataset' tab and inspect schema, distributions, or missingness rankings.", {}


def verify_mission_2(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 2: Verify cleaning without data loss (applied transformation with recorded What Changed)."""
    mgr = session_state.get("datalab_working_manager") or session_state.get("cleaning_manager")
    if mgr and len(mgr.history) > 0 and mgr.last_comparison:
        comp = mgr.last_comparison
        return True, f"Verified: Applied '{comp['step_info'].get('operation')}' resulting in {comp['rows_after']} rows ({comp['rows_delta']} rows delta).", {
            "operation": comp["step_info"].get("operation"),
            "rows_before": comp["rows_before"],
            "rows_after": comp["rows_after"],
            "missing_cells_delta": comp["missing_cells_delta"]
        }

    # Check Workflow Studio for executed cleaning step
    wf_store = session_state.get("workflow_studio_pipelines", {})
    for wf in wf_store.values():
        for step in wf.steps.values():
            if step.status == "completed" and step.what_changed:
                diff = step.what_changed
                return True, f"Verified via Workflow Studio: Executed '{step.name}' with verified What Changed delta.", {
                    "operation": step.op_type,
                    "rows_before": diff["rows_before"],
                    "rows_after": diff["rows_after"]
                }

    return False, "Incomplete: Apply a cleaning operation (e.g. Imputation, Row Filter, Deduplication) in 'Clean & Transform' or 'Workflow Studio'.", {}


def verify_mission_3(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 3: Verify feature engineering (derived new column)."""
    mgr = session_state.get("datalab_working_manager") or session_state.get("cleaning_manager")
    if mgr and len(mgr.history) > 0:
        for step in mgr.history:
            op_t = step.get("op_type", "")
            if op_t.startswith("feature_") or op_t in ("netflix_recipe", "numeric_binning", "datetime_extract", "text_transform"):
                return True, f"Verified: Engineered feature via '{op_t}' with configuration {step.get('params')}.", {
                    "feature_op": op_t,
                    "params": step.get("params")
                }

    wf_store = session_state.get("workflow_studio_pipelines", {})
    for wf in wf_store.values():
        for step in wf.steps.values():
            if step.status == "completed" and (step.op_type.startswith("feature_") or step.op_type == "netflix_recipe"):
                return True, f"Verified via Workflow Studio: Engineered feature via step '{step.name}'.", {
                    "feature_op": step.op_type
                }

    return False, "Incomplete: Create a derived feature (Binning, Math transform, Arithmetic, or Text length) in 'Clean & Transform' or 'Workflow Studio'.", {}


def verify_mission_4(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 4: Verify baseline and candidate model training."""
    last_res = session_state.get("model_lab_last_results")
    if last_res and len(last_res) >= 2:
        names = [r["model_name"] for r in last_res]
        has_baseline = "baseline" in names
        candidate_count = len([n for n in names if n != "baseline"])
        if has_baseline and candidate_count >= 1:
            primary_metric = "macro_f1" if last_res[0]["task_type"] == "classification" else "r2"
            base_score = next(r["metrics"].get(primary_metric) for r in last_res if r["model_name"] == "baseline")
            cand_score = next(r["metrics"].get(primary_metric) for r in last_res if r["model_name"] != "baseline")
            return True, f"Verified: Evaluated Baseline ({base_score}) against candidate model ({cand_score}) on held-out test split.", {
                "models_evaluated": names,
                "primary_metric": primary_metric,
                "baseline_score": base_score,
                "candidate_score": cand_score
            }

    wf_store = session_state.get("workflow_studio_pipelines", {})
    for wf in wf_store.values():
        for step in wf.steps.values():
            if step.status == "completed" and step.op_type == "ml_supervised_train" and step.output_model_result:
                return True, f"Verified via Workflow Studio: Fitted model '{step.output_model_result['model_name']}' with recorded test metrics.", {
                    "model_name": step.output_model_result["model_name"],
                    "metrics": step.output_model_result["metrics"]
                }

    return False, "Incomplete: Train a baseline model and at least one candidate estimator in 'Model Lab' Mode 1 or via Workflow Studio.", {}


def verify_mission_5(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 5: Diagnose model performance (inspected confusion matrix or residuals)."""
    last_res = session_state.get("model_lab_last_results")
    test_data = session_state.get("model_lab_test_data")
    if last_res and test_data:
        chosen = last_res[0]
        has_cm = "confusion_matrix" in chosen.get("extra", {})
        has_res = "residuals" in chosen.get("extra", {})
        if has_cm or has_res:
            return True, "Verified: Inspected held-out prediction diagnostic distribution and error patterns.", {
                "diagnostic_type": "confusion_matrix" if has_cm else "residuals",
                "sample_evaluated": len(test_data[1])
            }
    return False, "Incomplete: Run a model training cycle in 'Model Lab' to generate confusion matrix or residual error diagnostics.", {}


def verify_mission_6(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 6: Experiment with recommendation weights or feedback."""
    rec_run = session_state.get("rec_current_run")
    rec_feedback = session_state.get("recommender_feedback_master", {})
    if rec_run and len(rec_run.get("rec_results", [])) > 0:
        fb_count = sum(len(judg) for judg in rec_feedback.values())
        return True, f"Verified: Computed top recommendations for '{rec_run['query_title']}' with {len(rec_run['text_fields'])} metadata fields ({fb_count} relevance feedback judgments recorded).", {
            "query_title": rec_run["query_title"],
            "fields": rec_run["text_fields"],
            "user_judgments_count": fb_count
        }

    # Or check if quick recommender was queried
    if "title" in df.columns and session_state.get("sb_quick_rec_title"):
        return True, f"Verified: Generated content recommendations for title '{session_state.get('sb_quick_rec_title')}'.", {
            "query_title": session_state.get("sb_quick_rec_title")
        }

    return False, "Incomplete: Run a recommender experiment in 'Model Lab' (Mode 2 or Mode 3) and evaluate recommendation rankings.", {}


def verify_mission_7(df: pd.DataFrame, session_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Mission 7: Build and reproduce an end-to-end workflow."""
    wf_store = session_state.get("workflow_studio_pipelines", {})
    for wf in wf_store.values():
        if len(wf.step_order) >= 2:
            completed_steps = [s_id for s_id in wf.step_order if wf.steps[s_id].status == "completed"]
            if len(completed_steps) >= 2:
                return True, f"Verified: Workflow '{wf.name}' executed with {len(completed_steps)} completed steps and deterministic audit lineage.", {
                    "workflow_name": wf.name,
                    "pipeline_id": wf.pipeline_id,
                    "completed_steps": len(completed_steps)
                }

    return False, "Incomplete: Assemble and execute a workflow with at least 2 steps in 'Workflow Studio'.", {}


# =====================================================================
# EXTENSIBLE MISSION REGISTRY
# =====================================================================

MISSION_REGISTRY: Dict[str, DataScientistMission] = {
    "mission_1": DataScientistMission(
        mission_id="mission_1",
        title="Mission 1: Investigate the Dataset",
        difficulty="Beginner",
        objective="Inspect dimensions, data types, missingness rates, and distributions to establish a data quality baseline before modeling.",
        steps=[
            "1. Navigate to the 'Explore Dataset' tab.",
            "2. Inspect the overall row and column dimensions.",
            "3. Examine the missingness rankings and identifying fields with high null rates.",
            "4. Observe data type classifications (numeric, categorical, datetime, text)."
        ],
        hints={
            1: "Exploratory Data Analysis (EDA) is always the first defense against garbage-in, garbage-out data pipelines.",
            2: "In streaming catalogs, check whether TV Shows vs Movies exhibit different missingness patterns in key creative credits.",
            3: "Open 'Explore Dataset' > 'Missingness Audit' to view columns sorted by percentage missing."
        },
        verification_fn=verify_mission_1,
        reflection_question={
            "question": "Why is it dangerous to immediately drop all rows with missing values during dataset investigation?",
            "options": [
                "A) Missing values can be structural rather than accidental, and dropping rows can bias the sample.",
                "B) Pandas will throw an uncatchable syntax error if any row is dropped.",
                "C) Dropping rows automatically increases the random seed of subsequent models.",
                "D) Missing data never affects downstream model performance."
            ],
            "correct_idx": 0,
            "explanation": "Missingness often correlates with underlying variables (e.g. episodic TV shows lacking single directors). Dropping missing rows can destroy entire business segments from your training set."
        },
        tradeoff_explanation="Data quality issues should be diagnosed structurally. Distinguish between random missingness (MCAR) and systematic format constraints (MAR/MNAR).",
        badge="🔍 Catalog Detective",
        next_mission_id="mission_2"
    ),

    "mission_2": DataScientistMission(
        mission_id="mission_2",
        title="Mission 2: Clean Without Losing Useful Data",
        difficulty="Beginner",
        objective="Address missing values or duplicate records while preserving overall sample size and inspecting the exact consequences in What Changed?",
        steps=[
            "1. Open 'Clean & Transform' or 'Workflow Studio'.",
            "2. Choose an imputation strategy (Median for numeric, Mode or Constant for categoricals) or duplicate removal.",
            "3. Preview the operation to inspect the 'What Changed?' scorecard.",
            "4. Apply the operation and verify that the working copy updates deterministically."
        ],
        hints={
            1: "Median imputation is resilient to extreme outliers, whereas mean imputation shifts toward the tail.",
            2: "For categorical columns, constant fill (e.g. 'Unlisted') preserves missingness explicitly without making false assumptions.",
            3: "In 'Clean & Transform', select 'Impute Missing' > strategy 'median' and click 'Commit Transformation'."
        },
        verification_fn=verify_mission_2,
        reflection_question={
            "question": "What is the primary statistical trade-off when imputing missing values with the median?",
            "options": [
                "A) Median imputation preserves sample size but artificially compresses the feature's variance.",
                "B) Median imputation doubles the number of outliers in the dataset.",
                "C) Median imputation causes Scikit-learn pipelines to fail with sparse errors.",
                "D) Median imputation converts numeric columns into unstructured text."
            ],
            "correct_idx": 0,
            "explanation": "Replacing missing values with a central tendency estimate retains sample rows for training, but artificially clusters values at the center, reducing the observed variance."
        },
        tradeoff_explanation="Preserving rows through imputation trades variance accuracy for sample size. There is no one-size-fits-all strategy.",
        badge="🧹 Hygiene Architect",
        next_mission_id="mission_3"
    ),

    "mission_3": DataScientistMission(
        mission_id="mission_3",
        title="Mission 3: Engineer a Useful Feature",
        difficulty="Intermediate",
        objective="Create a derived feature (numeric binning, arithmetic combination, datetime extraction, or text length) to expose non-linear signals.",
        steps=[
            "1. In 'Clean & Transform' or 'Workflow Studio', select a feature engineering operation.",
            "2. Select a valid input column and configure parameters (e.g. bin quantiles or datetime component).",
            "3. Set an intuitive output column name.",
            "4. Preview and commit the transformation, verifying new columns in What Changed?"
        ],
        hints={
            1: "Raw continuous features often have non-linear effects; binning into quartiles lets linear models capture non-monotonic patterns.",
            2: "Text lengths and word counts serve as lightweight information-density proxies without needing expensive NLP vectorizers.",
            3: "Choose 'Feature Binning / Discretization', select a numeric column, set 4 bins, and click 'Commit Transformation'."
        },
        verification_fn=verify_mission_3,
        reflection_question={
            "question": "Why is feature engineering often more effective at boosting tabular model accuracy than tuning hyperparameters?",
            "options": [
                "A) Feature engineering exposes domain-relevant structural signals directly to the learning algorithm.",
                "B) Hyperparameter tuning is mathematically prohibited on Scikit-learn pipelines.",
                "C) Feature engineering eliminates the need for a train/test split.",
                "D) Tree models cannot learn without manually created interaction columns."
            ],
            "correct_idx": 0,
            "explanation": "Algorithms can only find relationships in the representations they receive. Exposing clear business signals (ratios, intervals, bins) often provides far greater leverage than marginal parameter tuning."
        },
        tradeoff_explanation="Derived features improve expressiveness but increase dimensionality and multicollinearity risk. Monitor feature count.",
        badge="⚙️ Feature Alchemist",
        next_mission_id="mission_4"
    ),

    "mission_4": DataScientistMission(
        mission_id="mission_4",
        title="Mission 4: Train a Baseline Model",
        difficulty="Intermediate",
        objective="Fit a Dummy baseline model alongside at least one candidate estimator, evaluating on held-out test splits without data leakage.",
        steps=[
            "1. Navigate to 'Model Lab' > 'Predictive Modeling Studio' (Mode 1).",
            "2. Select a target column and appropriate task type (Classification vs Regression).",
            "3. Check 'Baseline (Dummy)' and at least one candidate estimator (Logistic/Ridge, Decision Tree, Random Forest).",
            "4. Click 'Train & Compare Models' to benchmark candidate test metrics against the naive baseline."
        ],
        hints={
            1: "A 90% classification accuracy is completely unimpressive if the majority class represents 90% of the dataset! Always check the baseline.",
            2: "Ensure preprocessing (scaling and imputation) is fit strictly on training partitions to prevent test set data leakage.",
            3: "In Mode 1, pick target column, select 3-5 features, and click 'Train & Compare Models'."
        },
        verification_fn=verify_mission_4,
        reflection_question={
            "question": "Why must a machine learning model's performance always be compared against a naive baseline?",
            "options": [
                "A) To establish whether the learned model extracts genuine signal beyond class imbalance or mean guessing.",
                "B) Because Scikit-learn will crash if a baseline estimator is not included in the Python session.",
                "C) To guarantee that test set accuracy exceeds 99%.",
                "D) Baselines are only required for deep neural networks, not tabular models."
            ],
            "correct_idx": 0,
            "explanation": "On imbalanced datasets, predicting the majority class can yield high raw accuracy while having zero predictive power. Comparing against a dummy baseline proves actual predictive utility."
        },
        tradeoff_explanation="High test metrics on imbalanced data can be deceptive. Balanced Accuracy and Macro-F1 reveal whether minority classes are actually being learned.",
        badge="🎯 Benchmark Champion",
        next_mission_id="mission_5"
    ),

    "mission_5": DataScientistMission(
        mission_id="mission_5",
        title="Mission 5: Diagnose Model Performance",
        difficulty="Advanced",
        objective="Inspect a confusion matrix or residual error distribution to identify systematic failure modes rather than relying on a single aggregate metric.",
        steps=[
            "1. Train a model in 'Model Lab' Mode 1.",
            "2. Scroll down to 'Detailed Inspection & Error Analysis'.",
            "3. Select a candidate model to inspect its Confusion Matrix or Residuals Histogram.",
            "4. Review the Top Model Errors table to inspect specific misclassified or high-residual records."
        ],
        hints={
            1: "Aggregate metrics hide whether errors are random noise or concentrated in specific subgroups or classes.",
            2: "In regression, look for whether residuals are centered at zero with constant variance (homoscedasticity).",
            3: "In classification, look at the off-diagonal cells of the confusion matrix to identify which classes are confused."
        },
        verification_fn=verify_mission_5,
        reflection_question={
            "question": "What is the primary danger of optimizing purely for a single top-line metric (e.g. raw accuracy)?",
            "options": [
                "A) The model may sacrifice performance on rare or high-cost failure modes while appearing successful overall.",
                "B) Plotly charts cannot render more than one metric simultaneously.",
                "C) Single metrics cause memory leaks in Streamlit session state.",
                "D) Cross-validation folds become negative when evaluating multiple metrics."
            ],
            "correct_idx": 0,
            "explanation": "In real-world data science, different errors carry different business costs (e.g. false negatives in churn or fraud). Error diagnostics reveal where the model fails."
        },
        tradeoff_explanation="Error diagnosis moves data science from passive metric chasing to active debugging. Inspect individual misclassifications.",
        badge="🔬 Model Clinician",
        next_mission_id="mission_6"
    ),

    "mission_6": DataScientistMission(
        mission_id="mission_6",
        title="Mission 6: Experiment With Recommendations",
        difficulty="Intermediate",
        objective="Query content-based recommendations, alter field weights, evaluate ranking shifts, and understand the difference between similarity and relevance.",
        steps=[
            "1. Navigate to 'Model Lab' > 'Recommender Experiment Arena' (Mode 2) or 'Quick Netflix Recommendations' (Mode 3).",
            "2. Select a query item and examine its top-K similar matches.",
            "3. In Mode 2, adjust metadata field weights or ablate a field, re-run, and observe A/B rank shifts.",
            "4. Mark items as relevant/irrelevant in User Relevance Feedback to inspect judged ranking rates."
        ],
        hints={
            1: "Cosine similarity measures geometric proximity in high-dimensional term space, NOT whether a user will like the recommendation!",
            2: "Ablating (zero-weighting) a field isolates which metadata columns drive specific recommendation matches.",
            3: "Select a query title (e.g. 'Inception' or 'Stranger Things') and click 'Run Recommendation Experiment'."
        },
        verification_fn=verify_mission_6,
        reflection_question={
            "question": "Why is high cosine similarity between two item vectors NOT equivalent to high recommendation accuracy?",
            "options": [
                "A) High similarity only measures shared metadata terms, not true user preference, serendipity, or quality.",
                "B) Cosine similarity is always measured in radians rather than percentages.",
                "C) Cosine similarity only works on numerical spreadsheets, not text.",
                "D) Scikit-learn TF-IDF matrices discard all title information."
            ],
            "correct_idx": 0,
            "explanation": "Two movies may share keywords (e.g. 'space', 'action') yet appeal to completely different audiences. Content similarity is a ranking heuristic, not ground truth relevance."
        },
        tradeoff_explanation="Term similarity can produce redundant or repetitive recommendations without user relevance feedback. Balance similarity with diversity.",
        badge="🎬 Recommender Strategist",
        required_columns=["title"],
        next_mission_id="mission_7"
    ),

    "mission_7": DataScientistMission(
        mission_id="mission_7",
        title="Mission 7: Build and Reproduce a Workflow",
        difficulty="Advanced",
        objective="Assemble a multi-step pipeline in Workflow Studio, execute it deterministically, inspect intermediate states, and export the audit report.",
        steps=[
            "1. Open the 'Workflow Studio' tab.",
            "2. Select a starter workflow or assemble at least two connected steps (e.g. Deduplicate → Impute → Train Model).",
            "3. Click 'Run Complete Workflow' and monitor step execution statuses.",
            "4. Inspect intermediate outputs in 'What Changed?' and export the Markdown audit report."
        ],
        hints={
            1: "Reproducible pipelines guarantee that experiments can be audited, re-run, and verified without relying on hidden mutable state.",
            2: "If you modify an upstream step, Workflow Studio automatically invalidates dependent downstream nodes to prevent stale results.",
            3: "Click 'Run Complete Workflow' on the canvas and observe all step cards transition to 'COMPLETED'."
        },
        verification_fn=verify_mission_7,
        reflection_question={
            "question": "Why is deterministic DAG execution essential for production machine learning systems?",
            "options": [
                "A) It guarantees reproducible results, isolates step failures, and prevents hidden mutable state contamination.",
                "B) It allows Python to execute code without installing any dependencies.",
                "C) DAG workflows run exclusively on quantum computers.",
                "D) It ensures all machine learning models achieve 100% test accuracy."
            ],
            "correct_idx": 0,
            "explanation": "Data science in production fails when results cannot be traced back to exact input states and transformations. DAG execution guarantees reproducibility and auditability."
        },
        tradeoff_explanation="Pipeline automation enforces discipline and reproducibility. Versioned workflows turn ad-hoc analysis into reliable software.",
        badge="🏆 Master Data Scientist",
        next_mission_id=None
    ),
}


def get_mission(mission_id: str) -> Optional[DataScientistMission]:
    """Retrieves a mission by ID."""
    return MISSION_REGISTRY.get(mission_id)


def list_all_missions() -> List[DataScientistMission]:
    """Returns the ordered track of all missions."""
    return list(MISSION_REGISTRY.values())


# Alias for compatibility
list_missions = list_all_missions
