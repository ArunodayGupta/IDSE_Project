"""
workflow_engine.py
Phase 4: Visual Workflow Studio Deterministic Pipeline Engine.

Provides:
1. WorkflowStep and WorkflowPipeline DAG models with strict validation.
2. Topological ordering and cycle detection.
3. Step execution, intermediate output caching, and failure isolation.
4. Downstream invalidation propagation upon node modification or dataset change.
5. Reusable step execution leveraging Phase 2 (clean/transform) and Phase 3 (model/recommender) engines.
6. Honest "What Changed?" audit records (data differences vs model scorecard).
7. Versioned JSON serialization, safe allowlisted import, and starter workflows.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import copy
import json
import time
import math
import uuid
import pandas as pd
import numpy as np

from .operation_registry import get_operation_spec, OPERATION_REGISTRY
from .cleaning_lab import (
    WorkingDatasetManager,
    compute_what_changed,
    check_column_collisions,
    get_expected_output_columns,
)
from .model_lab import (
    validate_modeling_inputs,
    train_and_evaluate_model,
    RecommenderArenaService,
    compute_model_config_fingerprint,
)


class WorkflowExecutionResult(tuple):
    """Execution result that unpacks as (overall_success, exec_log)
    while also supporting dict-like key access result['success'] and result['log']."""
    def __new__(cls, success: bool, log: List[Dict[str, Any]]):
        return super().__new__(cls, (success, log))

    @property
    def success(self) -> bool:
        return self[0]

    @property
    def log(self) -> List[Dict[str, Any]]:
        return self[1]

    def __getitem__(self, item):
        if item == "success":
            return self[0]
        if item == "log":
            return self[1]
        return super().__getitem__(item)


class WorkflowStep:
    """Represents a single executable node within a WorkflowPipeline DAG."""

    def __init__(
        self,
        step_id: str,
        op_type: str,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        description: str = ""
    ):
        self.step_id = step_id
        self.op_type = op_type
        spec = get_operation_spec(op_type)
        self.name = name or (spec.name if spec else op_type.replace("_", " ").title())
        self.params = params or {}
        self.dependencies = dependencies or []
        self.description = description or (spec.description if spec else "")

        # Execution State
        self.status: str = "idle"  # idle | running | completed | failed | invalidated
        self.execution_duration: float = 0.0
        self.error_message: Optional[str] = None
        self.output_df: Optional[pd.DataFrame] = None
        self.output_model_result: Optional[Dict[str, Any]] = None
        self.what_changed: Optional[Dict[str, Any]] = None
        self.output_summary: Optional[Dict[str, Any]] = None

    @property
    def model_result(self) -> Optional[Dict[str, Any]]:
        return self.output_model_result

    def invalidate(self) -> None:
        """Invalidates this step's cached outputs."""
        self.status = "invalidated"
        self.output_df = None
        self.output_model_result = None
        self.what_changed = None
        self.output_summary = None
        self.error_message = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node configuration to pure dictionary (no DataFrame or code)."""
        return {
            "step_id": self.step_id,
            "op_type": self.op_type,
            "name": self.name,
            "params": copy.deepcopy(self.params),
            "dependencies": list(self.dependencies),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(
            step_id=data.get("step_id", str(uuid.uuid4())[:8]),
            op_type=data.get("op_type", "drop_missing_rows"),
            name=data.get("name"),
            params=data.get("params", {}),
            dependencies=data.get("dependencies", []),
            description=data.get("description", "")
        )


class WorkflowPipeline:
    """
    Deterministic Directed Acyclic Graph (DAG) Pipeline for DataLab Workflow Studio.
    """

    def __init__(
        self,
        name: str = "New Workflow",
        description: str = "",
        pipeline_id: Optional[str] = None,
        schema_version: str = "1.0.0"
    ):
        self.pipeline_id = pipeline_id or f"wf_{str(uuid.uuid4())[:8]}"
        self.name = name
        self.description = description
        self.schema_version = schema_version
        self.steps: Dict[str, WorkflowStep] = {}
        self.step_order: List[str] = []
        self.last_executed_at: Optional[str] = None

    def add_step(
        self,
        op_type: Any,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        insert_after: Optional[str] = None
    ) -> WorkflowStep:
        """Adds a new step node to the workflow."""
        if isinstance(op_type, WorkflowStep):
            step = op_type
            step_id = step.step_id
            self.steps[step_id] = step
            if step_id not in self.step_order:
                if insert_after and insert_after in self.step_order:
                    idx = self.step_order.index(insert_after) + 1
                    self.step_order.insert(idx, step_id)
                else:
                    self.step_order.append(step_id)
            self.invalidate_downstream(step_id)
            return step

        step_id = f"step_{len(self.steps) + 1}_{str(uuid.uuid4())[:4]}"
        deps = dependencies if dependencies is not None else ([self.step_order[-1]] if self.step_order else [])

        step = WorkflowStep(
            step_id=step_id,
            op_type=op_type,
            name=name,
            params=params or {},
            dependencies=deps
        )
        self.steps[step_id] = step

        if insert_after and insert_after in self.step_order:
            idx = self.step_order.index(insert_after) + 1
            self.step_order.insert(idx, step_id)
        else:
            self.step_order.append(step_id)

        self.invalidate_downstream(step_id)
        return step

    def remove_step(self, step_id: str) -> None:
        """Removes a step and cleans up dependent references."""
        if step_id not in self.steps:
            return

        self.invalidate_downstream(step_id)
        del self.steps[step_id]
        if step_id in self.step_order:
            self.step_order.remove(step_id)

        # Remove step_id from dependencies of remaining steps
        for s in self.steps.values():
            if step_id in s.dependencies:
                s.dependencies.remove(step_id)

    def move_step(self, step_id: str, direction: int) -> None:
        """Reorders a step up (-1) or down (+1) in the step sequence."""
        if step_id not in self.step_order:
            return
        idx = self.step_order.index(step_id)
        new_idx = idx + direction
        if 0 <= new_idx < len(self.step_order):
            self.step_order.pop(idx)
            self.step_order.insert(new_idx, step_id)
            self.invalidate_downstream(step_id)

    def update_step_params(self, step_id: str, new_params: Dict[str, Any], new_name: Optional[str] = None) -> None:
        """Updates node configuration and invalidates dependent outputs."""
        if step_id not in self.steps:
            return
        step = self.steps[step_id]
        step.params = copy.deepcopy(new_params)
        if new_name:
            step.name = new_name
        self.invalidate_downstream(step_id)

    def invalidate_downstream(self, step_id: str, include_self: bool = True) -> None:
        """Propagates invalidation to the step and all dependent downstream nodes."""
        to_invalidate: Set[str] = {step_id} if include_self else set()
        dep_set: Set[str] = {step_id}
        changed = True
        while changed:
            changed = False
            for s_id, step in self.steps.items():
                if s_id not in to_invalidate and s_id != step_id:
                    if any(dep in dep_set for dep in step.dependencies):
                        to_invalidate.add(s_id)
                        dep_set.add(s_id)
                        changed = True

        for s_id in to_invalidate:
            if s_id in self.steps:
                self.steps[s_id].invalidate()

    def get_step_output_df(self, step_id: str) -> Optional[pd.DataFrame]:
        """Retrieves the cached output DataFrame of a step if completed."""
        step = self.steps.get(step_id)
        return step.output_df if step else None

    def invalidate_all(self) -> None:
        """Invalidates all steps (e.g. when the input source dataset changes)."""
        for step in self.steps.values():
            step.invalidate()

    def get_topological_order(self) -> List[str]:
        """
        Validates graph acyclicity and returns step IDs in valid execution order.
        Raises ValueError if graph contains a cycle or missing dependency.
        """
        # Build adjacency graph
        in_degree: Dict[str, int] = {s_id: 0 for s_id in self.step_order}
        adj: Dict[str, List[str]] = {s_id: [] for s_id in self.step_order}

        for s_id in self.step_order:
            step = self.steps[s_id]
            for dep in step.dependencies:
                if dep not in self.steps:
                    raise ValueError(f"Step '{step.name}' ({s_id}) depends on non-existent step ID '{dep}'.")
                adj[dep].append(s_id)
                in_degree[s_id] += 1

        # Kahn's algorithm
        queue = [s_id for s_id in self.step_order if in_degree[s_id] == 0]
        order: List[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.step_order):
            raise ValueError("Cycle detected in workflow pipeline dependencies. Workflows must be strictly acyclic.")

        return order

    def validate_graph(self) -> Dict[str, Any]:
        """Validates graph acyclicity and dependency existence."""
        try:
            order = self.get_topological_order()
            return {"is_valid": True, "error": "", "order": order}
        except ValueError as e:
            return {"is_valid": False, "error": str(e), "order": []}

    get_execution_order = get_topological_order
    invalidate_from_step = invalidate_downstream

    def validate_pipeline(self, initial_df: Optional[pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validates the entire pipeline configuration against incoming data schema.
        """
        errors: List[str] = []
        try:
            order = self.get_topological_order()
        except ValueError as e:
            return False, [str(e)]

        if initial_df is None or len(initial_df) == 0:
            errors.append("Input dataset is empty (0 rows). Workflows require a valid non-empty dataset.")
            return False, errors

        # Dry-run simulate schema through steps
        current_cols = set(initial_df.columns)
        for s_id in order:
            step = self.steps[s_id]
            spec = get_operation_spec(step.op_type)
            if not spec:
                errors.append(f"Step '{step.name}': Unknown or unallowlisted operation '{step.op_type}'.")
                continue

            # Validate column presence in current simulated schema
            for p_name, p_spec in spec.params_schema.items():
                if p_spec.get("required") and p_spec.get("type") == "column":
                    col_val = step.params.get(p_name)
                    if not col_val or col_val not in current_cols:
                        errors.append(f"Step '{step.name}': Required input column '{col_val}' not present in preceding schema.")

            # Update simulated columns for feature engineering
            out_param = spec.output_col_param
            if out_param and step.params.get(out_param):
                new_col = str(step.params[out_param]).strip()
                current_cols.add(new_col)

        return (len(errors) == 0), errors

    def execute_step(
        self,
        step_id: str,
        initial_df: pd.DataFrame,
        dataset_identity: str = "dataset"
    ) -> bool:
        """
        Executes a single workflow step using cached results from its dependencies.
        Returns True if execution succeeded, False otherwise.
        """
        if step_id not in self.steps:
            return False

        step = self.steps[step_id]

        # Determine input DataFrame from immediate dependency or initial_df
        input_df = initial_df.copy()
        if step.dependencies:
            # Primary dependency is the last completed parent
            primary_dep_id = step.dependencies[-1]
            parent_step = self.steps.get(primary_dep_id)
            if parent_step and parent_step.status == "completed" and parent_step.output_df is not None:
                input_df = parent_step.output_df.copy()
            elif parent_step and parent_step.status != "completed":
                step.status = "failed"
                step.error_message = f"Cannot execute step: upstream dependency '{parent_step.name}' has not completed."
                return False

        step.status = "running"
        t0 = time.time()

        try:
            spec = get_operation_spec(step.op_type)
            if not spec:
                raise ValueError(f"Unknown operation type: '{step.op_type}'.")

            # Route 1: Supervised ML Step
            if step.op_type in ("ml_supervised_train", "train_supervised_model"):
                target_col = step.params.get("target_col")
                if not target_col or target_col not in input_df.columns:
                    raise ValueError(f"Target column '{target_col}' not found in input data.")

                feature_cols = step.params.get("feature_cols", [c for c in input_df.columns if c != target_col])
                task_type = step.params.get("task_type", "auto")
                if task_type == "auto":
                    task_type = "classification" if (input_df[target_col].nunique() < 10 or pd.api.types.is_object_dtype(input_df[target_col])) else "regression"

                modeling_df, meta = validate_modeling_inputs(
                    input_df, target_col=target_col, feature_cols=feature_cols, task_type=task_type
                )

                X = modeling_df[meta["usable_features"]]
                y = modeling_df[target_col]

                split_ratio = float(step.params.get("test_size", 0.2))
                seed_val = int(step.params.get("random_state", 42))
                cv_folds = int(step.params.get("cv_folds", 0))
                model_name = step.params.get("model_name", "logistic" if task_type == "classification" else "linear")

                stratify_y = None
                if task_type == "classification":
                    c_counts = y.value_counts()
                    if (c_counts >= 2).all() and len(y) * split_ratio >= len(c_counts):
                        stratify_y = y

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=split_ratio, random_state=seed_val, stratify=stratify_y
                )

                res = train_and_evaluate_model(
                    model_name=model_name,
                    task_type=task_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    numeric_cols=meta["numeric_features"],
                    categorical_cols=meta["categorical_features"],
                    random_state=seed_val,
                    cv_folds=cv_folds,
                    dataset_identity=dataset_identity
                )

                step.output_model_result = res
                step.output_df = input_df  # ML step does not mutate dataframe
                step.output_summary = {
                    "task_type": task_type,
                    "target_col": target_col,
                    "model_name": model_name,
                    "metrics": res["metrics"],
                    "fit_duration_sec": res["fit_duration_sec"],
                    "modeling_rows": len(X_train) + len(X_test),
                }

            # Route 2: Recommender Query Step
            elif step.op_type == "recommender_query":
                title_col = step.params.get("title_col", "title")
                if title_col not in input_df.columns:
                    raise ValueError(f"Item title column '{title_col}' not found.")

                query_title = step.params.get("query_title", "")
                text_fields = step.params.get("text_fields") or [
                    c for c in input_df.columns if c != title_col and (pd.api.types.is_object_dtype(input_df[c]) or pd.api.types.is_string_dtype(input_df[c]))
                ][:4]
                top_k = int(step.params.get("top_k", 5))

                arena = RecommenderArenaService(
                    df=input_df,
                    title_col=title_col,
                    text_fields=text_fields,
                    max_features=3000
                )
                recs, query_dur = arena.recommend(query_title, top_k=top_k)

                step.output_model_result = {
                    "query_title": query_title,
                    "top_k": top_k,
                    "recommendations": recs.to_dict(orient="records"),
                    "vocab_size": arena.vocabulary_size,
                    "query_duration_sec": query_dur,
                }
                step.output_df = recs
                step.output_summary = {
                    "query_title": query_title,
                    "vocab_size": arena.vocabulary_size,
                    "recs_count": len(recs),
                    "query_duration_sec": query_dur,
                }

            # Route 3: Pure Cleaning / Transformation / Feature Step (via WorkingDatasetManager)
            else:
                manager = WorkingDatasetManager(input_df, source_identity=dataset_identity)
                comp = manager.apply_operation(
                    op_type=step.op_type,
                    params=step.params,
                    treat_empty_as_missing=step.params.get("treat_empty_as_missing", True)
                )

                step.output_df = manager.get_working_df()
                step.what_changed = comp
                step.output_summary = {
                    "rows_before": comp["rows_before"],
                    "rows_after": comp["rows_after"],
                    "rows_delta": comp["rows_delta"],
                    "columns_added": comp["columns_added"],
                    "columns_removed": comp["columns_removed"],
                    "missing_cells_delta": comp["missing_cells_delta"],
                }

            step.status = "completed"
            step.error_message = None
            step.execution_duration = round(time.time() - t0, 4)
            return True

        except Exception as e:
            step.status = "failed"
            step.error_message = str(e)
            step.execution_duration = round(time.time() - t0, 4)
            self.invalidate_downstream(step_id, include_self=False)
            return False

    def execute_all(
        self,
        initial_df: pd.DataFrame,
        dataset_identity: str = "dataset"
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Executes all workflow steps in strict topological order from the beginning.
        """
        try:
            order = self.get_topological_order()
        except Exception as e:
            return False, [{"error": str(e)}]

        exec_log = []
        overall_success = True

        for s_id in order:
            step = self.steps[s_id]
            success = self.execute_step(s_id, initial_df=initial_df, dataset_identity=dataset_identity)
            exec_log.append({
                "step_id": s_id,
                "name": step.name,
                "status": step.status,
                "duration": step.execution_duration,
                "error": step.error_message
            })
            if not success:
                overall_success = False
                break

        self.last_executed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return WorkflowExecutionResult(overall_success, exec_log)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes workflow configuration to standard JSON dictionary."""
        return {
            "schema_version": self.schema_version,
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "step_order": list(self.step_order),
            "steps": [self.steps[s_id].to_dict() for s_id in self.step_order if s_id in self.steps],
            "last_executed_at": self.last_executed_at,
        }

    def to_json(self) -> str:
        """Exports workflow configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    export_json = to_json

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowPipeline":
        """Reconstructs and validates a workflow pipeline from dictionary."""
        version = data.get("schema_version", "1.0.0")
        pipe = cls(
            name=data.get("name", "Imported Workflow"),
            description=data.get("description", ""),
            pipeline_id=data.get("pipeline_id"),
            schema_version=version
        )

        steps_raw = data.get("steps", [])
        if not isinstance(steps_raw, list):
            raise ValueError("Invalid workflow schema: 'steps' must be a list.")

        for s_data in steps_raw:
            op_t = s_data.get("op_type")
            # Verify allowlisted operation
            if not get_operation_spec(op_t):
                raise ValueError(f"Unallowlisted or unsupported workflow operation: '{op_t}'.")

            step = WorkflowStep.from_dict(s_data)
            pipe.steps[step.step_id] = step

        # Restore step order
        declared_order = data.get("step_order", [])
        if declared_order and all(s_id in pipe.steps for s_id in declared_order):
            pipe.step_order = declared_order
        else:
            pipe.step_order = [s.step_id for s in steps_raw]

        # Verify acyclicity immediately upon import
        pipe.get_topological_order()
        return pipe

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowPipeline":
        """Parses and validates workflow from JSON string."""
        try:
            d = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        return cls.from_dict(d)

    import_json = from_json

    def export_audit_report(self) -> str:
        """Generates a comprehensive Markdown audit report for the workflow run."""
        lines = [
            f"# Workflow Studio Pipeline Execution Report: {self.name}",
            f"**Pipeline ID:** `{self.pipeline_id}`",
            f"**Execution Timestamp:** {self.last_executed_at or 'Never executed'}",
            f"**Total Steps:** {len(self.step_order)}",
            "",
            "## Pipeline Lineage & Execution Summary",
        ]

        if not self.step_order:
            lines.append("Workflow has no steps configured.")
        else:
            for idx, s_id in enumerate(self.step_order, 1):
                step = self.steps[s_id]
                status_emoji = "✅" if step.status == "completed" else ("❌" if step.status == "failed" else "⏳")
                lines.append(f"### Step {idx}: {step.name} {status_emoji}")
                lines.append(f"- **Operation:** `{step.op_type}`")
                lines.append(f"- **Status:** `{step.status.upper()}`")
                lines.append(f"- **Execution Duration:** {step.execution_duration}s")
                lines.append(f"- **Parameters:** `{step.params}`")

                if step.error_message:
                    lines.append(f"- **Error:** `{step.error_message}`")

                if step.output_summary:
                    lines.append("- **Output Summary:**")
                    for k, v in step.output_summary.items():
                        lines.append(f"  - *{k}:* {v}")
                lines.append("")

        return "\n".join(lines)


# =====================================================================
# USEFUL STARTER WORKFLOWS (PART C8)
# =====================================================================

def get_starter_workflows(
    df: Optional[pd.DataFrame] = None,
    dataset_name: str = "dataset"
) -> Dict[str, WorkflowPipeline]:
    """
    Constructs genuinely executable starter workflows adapted to available columns.
    """
    starters: Dict[str, WorkflowPipeline] = {}
    cols = list(df.columns) if df is not None else []

    # 1. Dataset Inspection & Profiling
    wf_inspect = WorkflowPipeline(
        name="Dataset Hygiene & Inspection",
        description="Standardizes whitespace to nulls and profiles missingness across records."
    )
    s1 = wf_inspect.add_step(
        op_type="replace_whitespace_with_nan",
        name="Standardize Whitespace Nulls",
        params={"columns": []}
    )
    s2 = wf_inspect.add_step(
        op_type="remove_duplicates",
        name="Deduplicate Rows",
        params={"keep": "first"},
        dependencies=[s1.step_id]
    )
    starters["inspection"] = wf_inspect

    # 2. Clean a Dataset
    wf_clean = WorkflowPipeline(
        name="Clean a Dataset",
        description="Detects duplicate rows, imputes numeric columns, and standardizes casing."
    )
    c1 = wf_clean.add_step(
        op_type="remove_duplicates",
        name="Remove Duplicates",
        params={"keep": "first"}
    )
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])] if df is not None else []
    if num_cols:
        c2 = wf_clean.add_step(
            op_type="impute_missing",
            name=f"Impute {num_cols[0]} (Median)",
            params={"column": num_cols[0], "strategy": "median"},
            dependencies=[c1.step_id]
        )
    str_cols = [c for c in cols if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])] if df is not None else []
    if str_cols:
        wf_clean.add_step(
            op_type="normalize_text",
            name=f"Normalize {str_cols[0]}",
            params={"column": str_cols[0], "case_style": "lower", "strip_whitespace": True},
            dependencies=[c1.step_id]
        )
    starters["clean"] = wf_clean

    # 3. Feature Engineering
    wf_feat = WorkflowPipeline(
        name="Feature Engineering Pipeline",
        description="Creates binned categories and text character lengths for downstream modeling."
    )
    if num_cols:
        f1 = wf_feat.add_step(
            op_type="feature_binning",
            name=f"Discretize {num_cols[0]}",
            params={"column": num_cols[0], "bin_type": "quantile", "bins": 4, "new_col_name": f"{num_cols[0]}_quartiles"}
        )
    if str_cols:
        wf_feat.add_step(
            op_type="feature_text_length",
            name=f"Measure {str_cols[0]} Length",
            params={"column": str_cols[0], "new_col_name": f"{str_cols[0]}_length"}
        )
    starters["feature_engineering"] = wf_feat

    # 4. Supervised Classification Workflow
    if cols:
        target_cand = None
        for c in cols:
            if df is not None and 2 <= df[c].nunique() <= 5:
                target_cand = c
                break
        target_cand = target_cand or cols[0]

        wf_class = WorkflowPipeline(
            name="Supervised Classification Workflow",
            description="Cleans missingness and fits a leakage-free Scikit-Learn classification model."
        )
        cl1 = wf_class.add_step(
            op_type="drop_missing_rows",
            name="Drop Missing Rows in Target",
            params={"columns": [target_cand], "how": "any"}
        )
        wf_class.add_step(
            op_type="ml_supervised_train",
            name=f"Train Classifier on {target_cand}",
            params={
                "target_col": target_cand,
                "task_type": "classification",
                "test_size": 0.2,
                "random_state": 42,
                "model_name": "logistic",
                "cv_folds": 3
            },
            dependencies=[cl1.step_id]
        )
        starters["classification"] = wf_class

    # 5. Supervised Regression Workflow
    if num_cols:
        wf_reg = WorkflowPipeline(
            name="Supervised Regression Workflow",
            description="Prepares numeric targets and fits a Ridge regression model with residual diagnostics."
        )
        reg_target = num_cols[-1]
        r1 = wf_reg.add_step(
            op_type="drop_missing_rows",
            name=f"Drop Nulls in {reg_target}",
            params={"columns": [reg_target], "how": "any"}
        )
        wf_reg.add_step(
            op_type="ml_supervised_train",
            name=f"Train Regressor on {reg_target}",
            params={
                "target_col": reg_target,
                "task_type": "regression",
                "test_size": 0.2,
                "random_state": 42,
                "model_name": "linear",
                "cv_folds": 3
            },
            dependencies=[r1.step_id]
        )
        starters["regression"] = wf_reg

    # 6. Netflix Recommendation Experiment (if 'title' column present)
    if "title" in cols:
        wf_rec = WorkflowPipeline(
            name="Netflix Recommendation Experiment",
            description="Engineers Netflix metadata soup and executes on-demand sparse recommender query."
        )
        rec1 = wf_rec.add_step(
            op_type="netflix_recipe",
            name="Engineer Metadata Soup",
            params={"recipe_name": "metadata_soup"}
        )
        q_title = "Stranger Things" if (df is not None and "Stranger Things" in df["title"].values) else (df["title"].iloc[0] if df is not None and not df.empty else "Inception")
        wf_rec.add_step(
            op_type="recommender_query",
            name="Query Recommendations",
            params={
                "title_col": "title",
                "query_title": q_title,
                "text_fields": [c for c in ["listed_in", "description", "director", "cast"] if c in cols],
                "top_k": 5
            },
            dependencies=[rec1.step_id]
        )
        starters["netflix_recommender"] = wf_rec

    return starters


# Aliases for compatibility
list_starter_workflows = get_starter_workflows
create_starter_workflow = get_starter_workflows
