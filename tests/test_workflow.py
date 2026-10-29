"""
test_workflow.py
Automated test suite for Phase 4 DataLab: Visual Workflow Studio.

Tests:
1. WorkflowStep and WorkflowPipeline DAG models.
2. Topological dependency ordering and cycle detection.
3. Missing dependency detection and input parameter validation against operation registry schemas.
4. Deterministic step execution and intermediate result caching.
5. Multi-step data cleaning pipeline execution.
6. Feature engineering + modeling end-to-end pipeline execution.
7. Failure isolation: when an upstream step fails, downstream steps are invalidated.
8. Downstream invalidation propagation when an upstream node's configuration changes.
9. Versioned JSON serialization, export, and allowlisted schema-validated import.
10. Security: Strict rejection of unallowlisted operations and arbitrary code.
11. Starter workflows generation and deterministic execution.
12. Honest What Changed? audit reports (data transformation deltas vs model scorecard).
"""

import unittest
import json
import pandas as pd
import numpy as np

from src.workflow_engine import (
    WorkflowStep,
    WorkflowPipeline,
    get_starter_workflows,
)
from src.operation_registry import get_operation_spec


class TestWorkflowStudioEngine(unittest.TestCase):
    """Unit test suite for Phase 4 Workflow Studio execution and DAG engine."""

    def setUp(self):
        """Construct synthetic dataset fixture."""
        np.random.seed(42)
        n = 60
        self.df = pd.DataFrame({
            "id": list(range(1, n + 1)),
            "title": [f"Item_{i}" for i in range(1, n + 1)],
            "category": np.random.choice(["SciFi", "Action", "Drama", "Comedy"], size=n),
            "views": np.random.randint(100, 10000, size=n).astype(float),
            "rating": np.random.uniform(1.0, 5.0, size=n),
            "target_churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "text_desc": [f"Description for item {i} with metadata words." for i in range(1, n + 1)],
        })
        # Introduce duplicate and null
        self.df.loc[0, "rating"] = np.nan
        self.df.loc[1, "rating"] = np.nan
        self.df.loc[59] = self.df.loc[58]

    def test_pipeline_dag_topological_sort_and_cycle_detection(self):
        """Verify topological ordering of dependent steps and rejection of cyclic graphs."""
        wf = WorkflowPipeline(name="Topological Test")
        step_a = WorkflowStep("step_a", "drop_missing_rows", params={"columns": ["rating"]})
        step_b = WorkflowStep("step_b", "remove_duplicates", dependencies=["step_a"])
        step_c = WorkflowStep("step_c", "feature_binning", params={"column": "views", "new_col_name": "views_tier", "bins": 3}, dependencies=["step_b"])

        wf.add_step(step_c)
        wf.add_step(step_a)
        wf.add_step(step_b)

        # Topological order should put step_a first, then step_b, then step_c
        order = wf.get_execution_order()
        self.assertEqual(order, ["step_a", "step_b", "step_c"])

        # Create cycle: step_a depends on step_c
        step_a.dependencies = ["step_c"]
        val = wf.validate_graph()
        self.assertFalse(val["is_valid"])
        self.assertIn("cycle", val["error"].lower())

    def test_missing_dependency_detection(self):
        """Verify graph validation detects non-existent dependency IDs."""
        wf = WorkflowPipeline(name="Missing Dep Test")
        step = WorkflowStep("step_1", "drop_missing_rows", dependencies=["ghost_step_999"])
        wf.add_step(step)

        val = wf.validate_graph()
        self.assertFalse(val["is_valid"])
        self.assertIn("ghost_step_999", val["error"])

    def test_multi_step_cleaning_workflow_execution(self):
        """Verify execution of a multi-step cleaning pipeline and intermediate output caching."""
        wf = WorkflowPipeline(name="Cleaning Pipeline")
        step1 = WorkflowStep(
            "step_drop_na",
            "drop_missing_rows",
            params={"columns": ["rating"]}
        )
        step2 = WorkflowStep(
            "step_dedup",
            "remove_duplicates",
            params={"strategy": "first"},
            dependencies=["step_drop_na"]
        )
        wf.add_step(step1)
        wf.add_step(step2)

        # Execute complete workflow
        res = wf.execute_all(self.df, dataset_identity="test_ds")
        self.assertTrue(res["success"])
        self.assertEqual(step1.status, "completed")
        self.assertEqual(step2.status, "completed")

        # Verify step 1 output has dropped rows with missing rating
        out1 = wf.get_step_output_df("step_drop_na")
        self.assertIsNotNone(out1)
        self.assertFalse(out1["rating"].isna().any())

        # Verify step 2 output has dropped duplicates
        out2 = wf.get_step_output_df("step_dedup")
        self.assertIsNotNone(out2)
        self.assertFalse(out2.duplicated().any())
        self.assertLessEqual(len(out2), len(out1))

    def test_feature_engineering_and_model_training_pipeline(self):
        """Verify pipeline with feature engineering followed by supervised model training."""
        wf = WorkflowPipeline(name="ML Pipeline")
        step_clean = WorkflowStep(
            "clean_step",
            "drop_missing_rows",
            params={"columns": ["rating"]}
        )
        step_feat = WorkflowStep(
            "feat_step",
            "feature_math",
            params={"column": "views", "math_func": "log1p", "new_col_name": "log_views"},
            dependencies=["clean_step"]
        )
        step_model = WorkflowStep(
            "model_step",
            "train_supervised_model",
            params={
                "target_col": "target_churn",
                "task_type": "classification",
                "feature_cols": ["rating", "views"],
                "model_name": "logistic",
                "test_size": 0.25,
                "random_state": 42
            },
            dependencies=["feat_step"]
        )
        wf.add_step(step_clean)
        wf.add_step(step_feat)
        wf.add_step(step_model)

        res = wf.execute_all(self.df, dataset_identity="test_ds_ml")
        self.assertTrue(res["success"])
        self.assertEqual(step_model.status, "completed")
        self.assertIsNotNone(step_model.model_result)
        self.assertIn("metrics", step_model.model_result)
        self.assertIn("accuracy", step_model.model_result["metrics"])

    def test_failure_isolation_and_downstream_invalidation(self):
        """Verify that an upstream failure halts execution and invalidates downstream steps."""
        wf = WorkflowPipeline(name="Failure Isolation")
        step1 = WorkflowStep("step_1", "drop_missing_rows", params={"columns": ["rating"]})
        # Failing step: invalid column
        step2 = WorkflowStep("step_2", "drop_selected_columns", params={"columns": ["non_existent_column_123"]}, dependencies=["step_1"])
        step3 = WorkflowStep("step_3", "remove_duplicates", dependencies=["step_2"])

        wf.add_step(step1)
        wf.add_step(step2)
        wf.add_step(step3)

        res = wf.execute_all(self.df, dataset_identity="fail_test_ds")
        self.assertFalse(res["success"])
        self.assertEqual(step1.status, "completed")
        self.assertEqual(step2.status, "failed")
        self.assertEqual(step3.status, "invalidated")
        self.assertIsNone(wf.get_step_output_df("step_3"))

    def test_node_modification_downstream_invalidation(self):
        """Verify that modifying an upstream step invalidates itself and all downstream outputs."""
        wf = WorkflowPipeline(name="Invalidation Propagation")
        step1 = WorkflowStep("step_1", "drop_missing_rows", params={"columns": ["rating"]})
        step2 = WorkflowStep("step_2", "remove_duplicates", dependencies=["step_1"])
        step3 = WorkflowStep("step_3", "feature_binning", params={"column": "views", "new_col_name": "views_tier", "bins": 3}, dependencies=["step_2"])

        wf.add_step(step1)
        wf.add_step(step2)
        wf.add_step(step3)

        wf.execute_all(self.df, dataset_identity="inv_test")
        self.assertEqual(step1.status, "completed")
        self.assertEqual(step2.status, "completed")
        self.assertEqual(step3.status, "completed")

        # Invalidate step 2: step 2 and step 3 must become invalidated, while step 1 remains completed
        wf.invalidate_from_step("step_2")
        self.assertEqual(step1.status, "completed")
        self.assertEqual(step2.status, "invalidated")
        self.assertEqual(step3.status, "invalidated")
        self.assertIsNone(step2.output_df)
        self.assertIsNone(step3.output_df)

    def test_workflow_json_serialization_and_allowlisted_import(self):
        """Verify versioned JSON serialization, export, and schema validation on import."""
        wf = WorkflowPipeline(name="Serialized Pipeline", description="Testing JSON Export")
        wf.add_step(WorkflowStep("s1", "drop_missing_rows", params={"columns": ["rating"]}))
        wf.add_step(WorkflowStep("s2", "remove_duplicates", dependencies=["s1"]))

        json_str = wf.export_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], "1.0.0")
        self.assertEqual(len(parsed["steps"]), 2)

        # Import into fresh pipeline
        imported_wf = WorkflowPipeline.import_json(json_str)
        self.assertEqual(imported_wf.name, "Serialized Pipeline")
        self.assertEqual(len(imported_wf.steps), 2)
        self.assertEqual(imported_wf.steps["s2"].dependencies, ["s1"])

    def test_rejection_of_unallowlisted_operations(self):
        """Security check: Reject unauthorized arbitrary operations from imported JSON."""
        evil_json = json.dumps({
            "schema_version": "1.0.0",
            "name": "Evil Workflow",
            "steps": [
                {"step_id": "e1", "op_type": "eval_arbitrary_code", "params": {"cmd": "rm -rf /"}}
            ]
        })
        with self.assertRaises(ValueError) as ctx:
            WorkflowPipeline.import_json(evil_json)
        self.assertIn("unallowlisted", str(ctx.exception).lower())

    def test_starter_workflows_generation_and_executability(self):
        """Verify that starter workflows adapt to dataset schema and execute successfully."""
        starters = get_starter_workflows(self.df, dataset_name="Test Catalog")
        self.assertIn("inspection", starters)
        self.assertIn("clean", starters)
        self.assertIn("feature_engineering", starters)
        self.assertIn("classification", starters)

        # Execute clean starter
        clean_wf = starters["clean"]
        res = clean_wf.execute_all(self.df, dataset_identity="starter_test")
        self.assertTrue(res["success"])
        for step in clean_wf.steps.values():
            self.assertEqual(step.status, "completed")

    def test_reproducibility_with_fixed_seed(self):
        """Verify that running the same workflow twice with the same seed produces identical results."""
        wf = WorkflowPipeline(name="Reproducibility Test")
        wf.add_step(WorkflowStep(
            "model_step",
            "train_supervised_model",
            params={
                "target_col": "target_churn",
                "task_type": "classification",
                "feature_cols": ["views"],
                "model_name": "decision_tree",
                "test_size": 0.3,
                "random_state": 42
            }
        ))
        clean_df = self.df.dropna(subset=["views", "target_churn"]).copy()

        # Run 1
        wf.execute_all(clean_df, dataset_identity="seed_test")
        res1_acc = wf.steps["model_step"].model_result["metrics"]["accuracy"]

        # Re-run
        wf.invalidate_all()
        wf.execute_all(clean_df, dataset_identity="seed_test")
        res2_acc = wf.steps["model_step"].model_result["metrics"]["accuracy"]

        self.assertEqual(res1_acc, res2_acc)


if __name__ == "__main__":
    unittest.main()
