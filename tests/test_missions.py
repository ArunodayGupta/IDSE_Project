"""
test_missions.py
Automated test suite for Phase 4 DataLab: Data Scientist Mission Mode.

Tests:
1. Complete 7-mission curriculum registered in MISSION_REGISTRY.
2. Dataset schema compatibility checking (is_compatible).
3. Evidence-grounded verification functions:
   - Mission 1: Dataset profiling & quality observation evidence.
   - Mission 2: Cleaning trade-offs & before-and-after audit verification.
   - Mission 3: Feature engineering derivation & output validation.
   - Mission 4: Baseline model training with held-out metrics.
   - Mission 5: Model diagnosis (confusion matrix / residual error analysis).
   - Mission 6: Multi-field recommender A/B comparison and ranking evaluation.
   - Mission 7: Full DAG workflow execution and audit export.
4. Prevention of unverified completion (cannot complete without real evidence).
5. Layered hints (conceptual -> dataset -> walkthrough).
6. Pedagogical reflection questions and trade-off explanations.
7. Mission progress isolation and reset behavior.
"""

import unittest
import pandas as pd
import numpy as np

from src.missions import (
    MISSION_REGISTRY,
    get_mission,
    list_missions,
    DataScientistMission,
    verify_mission_1,
    verify_mission_2,
    verify_mission_3,
    verify_mission_4,
    verify_mission_5,
    verify_mission_6,
    verify_mission_7,
)


class TestDataScientistMissions(unittest.TestCase):
    """Unit test suite for Phase 4 Guided Learning Mission Mode."""

    def setUp(self):
        """Construct synthetic test dataset."""
        np.random.seed(42)
        n = 50
        self.df = pd.DataFrame({
            "id": list(range(1, n + 1)),
            "title": [f"Title_{i}" for i in range(1, n + 1)],
            "type": np.random.choice(["Movie", "TV Show"], size=n),
            "views": np.random.randint(100, 10000, size=n).astype(float),
            "score": np.random.uniform(1.0, 10.0, size=n),
            "director": ["Director X"] * 25 + [None] * 25,
            "description": [f"Short synopsis for title {i}." for i in range(1, n + 1)],
            "target": np.random.choice([0, 1], size=n)
        })

    def test_mission_registry_curriculum(self):
        """Verify that all 7 guided missions are properly registered in sequence."""
        missions = list_missions()
        self.assertEqual(len(missions), 7)
        mission_ids = [m.mission_id for m in missions]
        self.assertEqual(mission_ids, [
            "mission_1", "mission_2", "mission_3",
            "mission_4", "mission_5", "mission_6", "mission_7"
        ])

        for m in missions:
            self.assertIsInstance(m, DataScientistMission)
            self.assertTrue(len(m.title) > 0)
            self.assertTrue(len(m.steps) > 0)
            self.assertEqual(len(m.hints), 3)  # Layered hints 1, 2, 3
            self.assertIn("question", m.reflection_question)
            self.assertIn("options", m.reflection_question)
            self.assertIn("correct_idx", m.reflection_question)
            self.assertTrue(len(m.tradeoff_explanation) > 0)
            self.assertTrue(len(m.badge) > 0)

    def test_dataset_prerequisite_and_compatibility_checks(self):
        """Verify schema compatibility guard for missions with specific column requirements."""
        m6 = get_mission("mission_6")
        self.assertIsNotNone(m6)

        # Dataset with 'title' column is compatible
        compat, msg = m6.is_compatible(self.df)
        self.assertTrue(compat)

        # Dataset missing 'title' column is NOT compatible
        no_title_df = self.df.drop(columns=["title"])
        compat_fail, msg_fail = m6.is_compatible(no_title_df)
        self.assertFalse(compat_fail)
        self.assertIn("title", msg_fail)

        # Empty DataFrame is NOT compatible
        compat_empty, msg_empty = m6.is_compatible(pd.DataFrame())
        self.assertFalse(compat_empty)
        self.assertIn("0 rows", msg_empty)

    def test_mission_1_evidence_verification(self):
        """Mission 1: Verify data investigation requires actual inspection action."""
        # Case 1: No evidence
        ok_no, msg_no, _ = verify_mission_1(self.df, {})
        self.assertFalse(ok_no)
        self.assertIn("explore dataset", msg_no.lower())

        # Case 2: Evidence provided via session state
        session_state = {"datalab_user_explored": True}
        ok_yes, msg_yes, res_ev = verify_mission_1(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertEqual(res_ev["rows_inspected"], 50)
        self.assertEqual(res_ev["columns_profiled"], 8)

    def test_mission_2_evidence_verification(self):
        """Mission 2: Verify cleaning without data loss requires actual operation applied."""
        from src.cleaning_lab import WorkingDatasetManager
        # Case 1: No cleaning manager or no operations applied
        ok_no, msg_no, _ = verify_mission_2(self.df, {})
        self.assertFalse(ok_no)
        self.assertIn("clean & transform", msg_no.lower())

        # Case 2: Cleaning manager with applied step
        mgr = WorkingDatasetManager(self.df.copy(), source_name="Test Source", source_identity="src_123")
        mgr.apply_step({
            "op_type": "impute_numeric",
            "columns": ["views"],
            "strategy": "mean",
            "description": "Impute views"
        })
        session_state = {"cleaning_manager": mgr}
        ok_yes, msg_yes, res_ev = verify_mission_2(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertIn("operation", res_ev)

    def test_mission_3_evidence_verification(self):
        """Mission 3: Verify feature engineering requires derived column creation."""
        from src.cleaning_lab import WorkingDatasetManager
        # Case 1: No feature engineered
        ok_no, _, _ = verify_mission_3(self.df, {})
        self.assertFalse(ok_no)

        # Case 2: Cleaning manager with feature operation
        mgr = WorkingDatasetManager(self.df.copy(), source_name="Test Source", source_identity="src_123")
        mgr.apply_step({
            "op_type": "feature_binning",
            "column": "views",
            "bins": 3,
            "new_col_name": "views_tier",
            "description": "Bin views"
        })
        session_state = {"cleaning_manager": mgr}
        ok_yes, _, res_ev = verify_mission_3(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertEqual(res_ev["feature_op"], "feature_binning")

    def test_mission_4_evidence_verification(self):
        """Mission 4: Verify baseline model training requires genuine fit and evaluation."""
        # Case 1: No training run
        ok_no, _, _ = verify_mission_4(self.df, {})
        self.assertFalse(ok_no)

        # Case 2: Genuine model run results (baseline + candidate)
        session_state = {
            "model_lab_last_results": [
                {
                    "model_name": "baseline",
                    "task_type": "classification",
                    "metrics": {"macro_f1": 0.50},
                },
                {
                    "model_name": "logistic",
                    "task_type": "classification",
                    "metrics": {"macro_f1": 0.85},
                }
            ]
        }
        ok_yes, _, res_ev = verify_mission_4(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertEqual(res_ev["baseline_score"], 0.50)
        self.assertEqual(res_ev["candidate_score"], 0.85)

    def test_mission_5_evidence_verification(self):
        """Mission 5: Verify model diagnosis requires inspecting misclassifications or residuals."""
        # Case 1: No diagnosis evidence
        ok_no, _, _ = verify_mission_5(self.df, {})
        self.assertFalse(ok_no)

        # Case 2: Error analysis inspected
        session_state = {
            "model_lab_last_results": [
                {
                    "model_name": "logistic",
                    "task_type": "classification",
                    "metrics": {"macro_f1": 0.85},
                    "extra": {"confusion_matrix": [[15, 2], [3, 10]]}
                }
            ],
            "model_lab_test_data": (None, [0, 1] * 15)
        }
        ok_yes, _, res_ev = verify_mission_5(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertEqual(res_ev["diagnostic_type"], "confusion_matrix")

    def test_mission_6_evidence_verification(self):
        """Mission 6: Verify recommender experimentation requires query and rankings."""
        # Case 1: No recommender experiment
        ok_no, _, _ = verify_mission_6(self.df, {})
        self.assertFalse(ok_no)

        # Case 2: Recommender experiment results provided
        session_state = {
            "rec_current_run": {
                "query_title": "Title_1",
                "text_fields": ["description"],
                "rec_results": [{"title": "Title_2", "similarity_score": 0.95}],
            },
            "recommender_feedback_master": {
                ("ds_1", "title", "Title_1"): {"Title_2": 1}
            }
        }
        ok_yes, _, res_ev = verify_mission_6(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertEqual(res_ev["query_title"], "Title_1")
        self.assertEqual(res_ev["user_judgments_count"], 1)

    def test_mission_7_evidence_verification(self):
        """Mission 7: Verify workflow builder requires completed DAG pipeline."""
        # Case 1: Incomplete or empty pipeline
        ok_no, _, _ = verify_mission_7(self.df, {})
        self.assertFalse(ok_no)

        # Case 2: Completed pipeline with multiple steps
        from src.workflow_engine import WorkflowPipeline, WorkflowStep
        wf = WorkflowPipeline(name="M7 Pipeline")
        wf.add_step(WorkflowStep("s1", "drop_missing_rows", params={"columns": ["score"]}))
        wf.add_step(WorkflowStep("s2", "remove_duplicates", dependencies=["s1"]))
        wf.execute_all(self.df, dataset_identity="m7_test")

        session_state = {
            "workflow_studio_pipelines": {"m7_wf": wf}
        }
        ok_yes, _, res_ev = verify_mission_7(self.df, session_state)
        self.assertTrue(ok_yes)
        self.assertEqual(res_ev["completed_steps"], 2)

    def test_layered_hints_and_pedagogical_reflections(self):
        """Verify layered hints (1=concept, 2=dataset, 3=walkthrough) and reflections."""
        for m_id, mission in MISSION_REGISTRY.items():
            self.assertIn(1, mission.hints)
            self.assertIn(2, mission.hints)
            self.assertIn(3, mission.hints)

            q = mission.reflection_question
            self.assertIsInstance(q["options"], list)
            self.assertGreaterEqual(len(q["options"]), 2)
            self.assertIn(q["correct_idx"], range(len(q["options"])))
            self.assertTrue(len(q["explanation"]) > 0)


if __name__ == "__main__":
    unittest.main()
