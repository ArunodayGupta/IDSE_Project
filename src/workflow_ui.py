"""
workflow_ui.py
Phase 4: Visual Workflow Studio Streamlit Interactive Interface.

Provides:
- Connected interactive canvas with operation cards and ordering controls.
- Node configuration drawer with schema validation and parameter forms.
- Re-execution, step preview, and failure isolation.
- Honest "What Changed?" data deltas vs model scorecards.
- Session save, duplication, JSON import/export, and Markdown audit export.
- Seamless execution against active DataLab source or working copy.
"""

from typing import Optional, Dict, Any, List
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from .data_context import get_source_dataset, get_working_dataset
from .operation_registry import list_operations_by_category, get_operation_spec
from .workflow_engine import WorkflowPipeline, WorkflowStep, get_starter_workflows
from src.ui_theme import (
    render_page_header,
    render_callout,
    render_badge,
    COLOR_CYAN,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_VIOLET
)


def render_workflow_studio_tab(netflix_raw_df: Optional[pd.DataFrame] = None) -> None:
    """
    Renders the Workflow Studio visual environment.
    """
    render_page_header(
        title="Workflow Studio: Visual Pipeline Builder",
        subtitle="Assemble, configure, validate, and execute end-to-end data science pipelines from reusable operations.",
        badge="DAG Engine",
        badge_type="info"
    )

    # ------------------------------------------------------------------
    # Step 1: Input Dataset Selection
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

    if source_df is None or source_df.empty:
        st.warning("⚠️ No dataset loaded. Please load or select a dataset in **DataLab Explore**.")
        return

    st.markdown("##### 📁 Workflow Input Dataset")
    ds_c1, ds_c2 = st.columns([2, 3])
    with ds_c1:
        data_source_mode = st.radio(
            "Select Initial Pipeline Input:",
            options=["Raw Source Snapshot", "Clean & Transform Working Copy"],
            index=0,
            horizontal=True,
            key="wf_ds_mode"
        )

    if data_source_mode == "Clean & Transform Working Copy" and working_df is not None:
        active_input_df = working_df
        active_dataset_id = f"{source_ident}::working::{working_state_id}"
    else:
        active_input_df = source_df
        active_dataset_id = f"{source_ident}::raw_source"

    is_input_empty = (active_input_df is None or len(active_input_df) == 0)

    with ds_c2:
        if is_input_empty:
            st.error(f"**Active Input:** `{source_name}` — **Empty (0 rows)**. Pipeline execution is disabled until data is restored.")
        else:
            status_tag = f"✨ Working Copy ({len(active_input_df):,} rows × {active_input_df.shape[1]} cols)" if data_source_mode == "Clean & Transform Working Copy" else f"📄 Raw Source ({len(active_input_df):,} rows × {active_input_df.shape[1]} cols)"
            st.info(f"**Pipeline Ingestion:** `{source_name}` — **{status_tag}** (ID: `{active_dataset_id[:24]}...`)")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Step 2: Workflow Management & Session State
    # ------------------------------------------------------------------
    workflows_store: Dict[str, WorkflowPipeline] = st.session_state.setdefault("workflow_studio_pipelines", {})
    active_wf_id = st.session_state.setdefault("workflow_studio_active_id", "")

    if not workflows_store:
        starters = get_starter_workflows(active_input_df, dataset_name=source_name)
        workflows_store.update(starters)
        active_wf_id = list(starters.keys())[0]
        st.session_state["workflow_studio_active_id"] = active_wf_id

    # Management Bar
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns([2, 1, 1, 1, 1])

    with m_col1:
        pipeline_options = list(workflows_store.keys())
        active_wf_id = st.selectbox(
            "Select Active Workflow:",
            options=pipeline_options,
            index=pipeline_options.index(active_wf_id) if active_wf_id in pipeline_options else 0,
            format_func=lambda x: f"{workflows_store[x].name} ({len(workflows_store[x].step_order)} steps)",
            key="sb_active_wf"
        )
        st.session_state["workflow_studio_active_id"] = active_wf_id

    current_pipeline = workflows_store[active_wf_id]

    with m_col2:
        st.write("")
        st.write("")
        if st.button("➕ New Blank", use_container_width=True):
            new_wf = WorkflowPipeline(name="Custom Workflow")
            workflows_store[new_wf.pipeline_id] = new_wf
            st.session_state["workflow_studio_active_id"] = new_wf.pipeline_id
            st.rerun()

    with m_col3:
        st.write("")
        st.write("")
        if st.button("📋 Duplicate", use_container_width=True):
            dup_dict = current_pipeline.to_dict()
            dup_dict["name"] = f"{current_pipeline.name} (Copy)"
            dup_dict["pipeline_id"] = f"wf_{len(workflows_store)+1}"
            dup_wf = WorkflowPipeline.from_dict(dup_dict)
            workflows_store[dup_wf.pipeline_id] = dup_wf
            st.session_state["workflow_studio_active_id"] = dup_wf.pipeline_id
            st.rerun()

    with m_col4:
        st.write("")
        st.write("")
        json_cfg = current_pipeline.to_json()
        st.download_button(
            "📥 Export JSON",
            data=json_cfg,
            file_name=f"{current_pipeline.name.lower().replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

    with m_col5:
        st.write("")
        st.write("")
        if len(workflows_store) > 1 and st.button("🗑️ Delete", use_container_width=True):
            del workflows_store[active_wf_id]
            st.session_state["workflow_studio_active_id"] = list(workflows_store.keys())[0]
            st.rerun()

    # JSON Import Accordion
    with st.expander("📤 Import Workflow from JSON"):
        uploaded_json = st.file_uploader("Upload Workflow JSON Schema:", type=["json"], key="wf_json_upload")
        if uploaded_json is not None:
            try:
                raw_text = uploaded_json.getvalue().decode("utf-8")
                imported_wf = WorkflowPipeline.from_json(raw_text)
                st.success(f"Validated workflow: **{imported_wf.name}** ({len(imported_wf.step_order)} steps).")
                if st.button("Confirm & Load Imported Workflow", key="btn_confirm_import"):
                    workflows_store[imported_wf.pipeline_id] = imported_wf
                    st.session_state["workflow_studio_active_id"] = imported_wf.pipeline_id
                    st.rerun()
            except Exception as e:
                st.error(f"Invalid workflow configuration: {str(e)}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Step 3: Interactive Visual Canvas & Palette
    # ------------------------------------------------------------------
    canvas_col, config_col = st.columns([7, 5])

    with canvas_col:
        st.markdown(f"### 🔗 Connected Workflow Canvas: *{current_pipeline.name}*")
        st.caption(current_pipeline.description or "Click 'Add Step' to extend the pipeline, configure node parameters, and execute.")

        # Action Execution Bar
        exec_c1, exec_c2, exec_c3 = st.columns([2, 2, 2])
        with exec_c1:
            run_all_btn = st.button("🚀 Run Complete Workflow", type="primary", use_container_width=True, disabled=is_input_empty)
        with exec_c2:
            is_valid, validation_errors = current_pipeline.validate_pipeline(active_input_df)
            if is_valid:
                st.success("✅ Graph Valid (Acyclic & Compatible)")
            else:
                st.error(f"⚠️ Validation Issues ({len(validation_errors)})")
        with exec_c3:
            report_md = current_pipeline.export_audit_report()
            st.download_button("📄 Download Audit Report", data=report_md, file_name=f"{current_pipeline.name}_audit.md", mime="text/markdown", use_container_width=True)

        if not is_valid:
            for err in validation_errors:
                st.caption(f"❌ {err}")

        # Execute Pipeline on trigger
        if run_all_btn:
            with st.spinner("Executing workflow steps in topological order..."):
                overall_ok, exec_log = current_pipeline.execute_all(active_input_df, dataset_identity=active_dataset_id)
                if overall_ok:
                    st.success("🎉 All workflow steps completed successfully!")
                else:
                    st.error("Workflow execution halted due to a step failure. See failed node below for details.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Step Cards Canvas
        if not current_pipeline.step_order:
            st.info("This workflow has no steps. Use the 'Add Step Node' palette on the right to build your pipeline.")
        else:
            for idx, s_id in enumerate(current_pipeline.step_order, 1):
                step = current_pipeline.steps[s_id]

                # Status Badge Styling
                status_color = "#6C757D"
                status_badge = "⚪ IDLE"
                if step.status == "completed":
                    status_color = "#00D97E"
                    status_badge = f"✅ COMPLETED ({step.execution_duration}s)"
                elif step.status == "failed":
                    status_color = "#E50914"
                    status_badge = "❌ FAILED"
                elif step.status == "running":
                    status_color = "#2C7BE5"
                    status_badge = "⏳ RUNNING"
                elif step.status == "invalidated":
                    status_color = "#FFA000"
                    status_badge = "🕒 STALE / INVALIDATED"

                # Render Step Card
                st.markdown(f"""
                    <div style="background: linear-gradient(180deg, rgba(20, 28, 46, 0.95) 0%, rgba(14, 20, 34, 0.95) 100%); border: 1px solid #1E2B44; border-left: 4px solid {status_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="background: rgba(0, 210, 255, 0.15); color: #00D2FF; border: 1px solid rgba(0, 210, 255, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">STEP #{idx}</span>
                                <span style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">{step.name}</span>
                            </div>
                            <span style="background: rgba(20, 28, 46, 0.9); color: {status_color}; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                                {status_badge}
                            </span>
                        </div>
                        <div style="color: #94A3B8; font-size: 0.82rem; margin: 6px 0;">
                            <b>Operation:</b> <code>{step.op_type}</code> &nbsp;|&nbsp;
                            <b>Dependencies:</b> {', '.join(step.dependencies) if step.dependencies else 'None (Root)'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Step Card Action Bar
                btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns([2, 1, 1, 1, 2])
                with btn_c1:
                    if st.button("⚙️ Configure", key=f"btn_cfg_{s_id}", use_container_width=True):
                        st.session_state["wf_selected_step_id"] = s_id
                        st.rerun()
                with btn_c2:
                    if idx > 1 and st.button("⬆️", key=f"btn_up_{s_id}", help="Move Step Up"):
                        current_pipeline.move_step(s_id, -1)
                        st.rerun()
                with btn_c3:
                    if idx < len(current_pipeline.step_order) and st.button("⬇️", key=f"btn_dn_{s_id}", help="Move Step Down"):
                        current_pipeline.move_step(s_id, 1)
                        st.rerun()
                with btn_c4:
                    if st.button("🗑️", key=f"btn_rm_{s_id}", help="Remove Step"):
                        current_pipeline.remove_step(s_id)
                        st.rerun()
                with btn_c5:
                    if st.button("▶ Run Step", key=f"btn_run_single_{s_id}", use_container_width=True, disabled=is_input_empty):
                        with st.spinner(f"Running step '{step.name}'..."):
                            current_pipeline.execute_step(s_id, initial_df=active_input_df, dataset_identity=active_dataset_id)
                            st.rerun()

                if step.error_message:
                    st.error(f"⚠️ Error: {step.error_message}")

                # Connector Arrow down to next step
                if idx < len(current_pipeline.step_order):
                    st.markdown("""
                        <div style="text-align: center; color: #ADB5BD; font-size: 1.1rem; margin: -6px 0 6px 0;">
                            ↓
                        </div>
                    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Step 4: Node Configuration & Palette Drawer
    # ------------------------------------------------------------------
    with config_col:
        selected_step_id = st.session_state.get("wf_selected_step_id")
        selected_step: Optional[WorkflowStep] = current_pipeline.steps.get(selected_step_id) if selected_step_id else None

        if selected_step:
            st.markdown(f"### ⚙️ Configure Step: *{selected_step.name}*")
            spec = get_operation_spec(selected_step.op_type)
            if spec:
                st.caption(f"**Category:** {spec.category} • *{spec.description}*")
                st.info(f"💡 **Data Science Principle:** {spec.explanation}")

            # Step Name Editor
            new_step_name = st.text_input("Step Name:", value=selected_step.name, key=f"inp_sname_{selected_step.step_id}")

            # Node Parameter Form
            new_params: Dict[str, Any] = {}
            if spec:
                # Determine columns available from previous step or active input
                available_cols = list(active_input_df.columns)
                if selected_step.dependencies:
                    p_id = selected_step.dependencies[-1]
                    p_step = current_pipeline.steps.get(p_id)
                    if p_step and p_step.output_df is not None:
                        available_cols = list(p_step.output_df.columns)

                for p_name, p_schema in spec.params_schema.items():
                    p_type = p_schema.get("type")
                    label = p_schema.get("label", p_name)
                    cur_val = selected_step.params.get(p_name, p_schema.get("default"))

                    if p_type == "column":
                        def_idx = available_cols.index(cur_val) if cur_val in available_cols else 0
                        new_params[p_name] = st.selectbox(label, options=available_cols, index=def_idx, key=f"p_{p_name}_{selected_step.step_id}")
                    elif p_type == "multiselect":
                        def_sel = [c for c in (cur_val or []) if c in available_cols]
                        new_params[p_name] = st.multiselect(label, options=available_cols, default=def_sel, key=f"p_{p_name}_{selected_step.step_id}")
                    elif p_type == "select":
                        opts = p_schema.get("options", [])
                        def_idx = opts.index(cur_val) if cur_val in opts else 0
                        new_params[p_name] = st.selectbox(label, options=opts, index=def_idx, key=f"p_{p_name}_{selected_step.step_id}")
                    elif p_type == "bool":
                        new_params[p_name] = st.checkbox(label, value=bool(cur_val), key=f"p_{p_name}_{selected_step.step_id}")
                    elif p_type == "int":
                        new_params[p_name] = st.number_input(label, value=int(cur_val or 0), step=1, key=f"p_{p_name}_{selected_step.step_id}")
                    elif p_type == "float":
                        new_params[p_name] = st.number_input(label, value=float(cur_val or 0.0), step=0.05, key=f"p_{p_name}_{selected_step.step_id}")
                    else:
                        new_params[p_name] = st.text_input(label, value=str(cur_val or ""), key=f"p_{p_name}_{selected_step.step_id}")

            # Dependency Selector
            available_deps = [s_id for s_id in current_pipeline.step_order if s_id != selected_step.step_id]
            chosen_deps = st.multiselect(
                "Upstream Dependencies:",
                options=available_deps,
                default=[d for d in selected_step.dependencies if d in available_deps],
                format_func=lambda x: current_pipeline.steps[x].name if x in current_pipeline.steps else x,
                key=f"ms_deps_{selected_step.step_id}"
            )

            if st.button("💾 Apply Configuration & Invalidate Downstream", type="primary", use_container_width=True):
                selected_step.dependencies = chosen_deps
                current_pipeline.update_step_params(selected_step.step_id, new_params=new_params, new_name=new_step_name)
                st.success("Configuration applied! Dependent downstream nodes marked for re-execution.")
                st.rerun()

            st.markdown("---")

        # --------------------------------------------------------------
        # Operation Palette: Add Step Node
        # --------------------------------------------------------------
        st.markdown("### ➕ Add Step to Workflow")
        st.caption("Select an operation from the categorized DataLab operation registry to append to this workflow:")

        categorized_ops = list_operations_by_category()
        selected_cat = st.selectbox("Operation Category:", options=list(categorized_ops.keys()), key="sb_op_category")

        cat_specs = categorized_ops.get(selected_cat, [])
        selected_op_spec = st.selectbox(
            "Select Operation:",
            options=cat_specs,
            format_func=lambda s: f"{s.name} — {s.description[:40]}...",
            key="sb_palette_op"
        )

        if selected_op_spec:
            st.caption(f"**Action:** {selected_op_spec.description}")

            # Quick-configure default params based on available columns
            default_p = {}
            for k, s_p in selected_op_spec.params_schema.items():
                if s_p.get("type") == "column" and active_input_df.columns.size > 0:
                    default_p[k] = active_input_df.columns[0]
                else:
                    default_p[k] = s_p.get("default")

            if st.button(f"Add '{selected_op_spec.name}' Node", use_container_width=True):
                new_s = current_pipeline.add_step(
                    op_type=selected_op_spec.op_type,
                    params=default_p,
                    name=selected_op_spec.name
                )
                st.session_state["wf_selected_step_id"] = new_s.step_id
                st.success(f"Added '{selected_op_spec.name}' to workflow!")
                st.rerun()

    # ------------------------------------------------------------------
    # Step 5: What Changed? / Output Inspection
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔬 Step Output & 'What Changed?' Inspector")
    st.caption("Inspect the actual data deltas, new columns, or model metrics produced by executed steps.")

    completed_steps = [current_pipeline.steps[s_id] for s_id in current_pipeline.step_order if current_pipeline.steps[s_id].status == "completed"]

    if not completed_steps:
        st.info("Run steps in your workflow above to inspect intermediate data transformations, feature metrics, and modeling scorecards.")
    else:
        inspect_step_id = st.selectbox(
            "Select Completed Step to Inspect:",
            options=[s.step_id for s in completed_steps],
            format_func=lambda x: f"{current_pipeline.steps[x].name} ({current_pipeline.steps[x].op_type})",
            key="sb_inspect_step"
        )
        inspected_step = current_pipeline.steps[inspect_step_id]

        # Case A: Supervised ML Step Scorecard (Defect C6)
        if inspected_step.op_type == "ml_supervised_train" and inspected_step.output_model_result:
            st.markdown("#### 🎯 Machine Learning Model Scorecard")
            m_res = inspected_step.output_model_result
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", m_res["model_name"].title())
            m_col2.metric("Task", m_res["task_type"].title())
            m_col3.metric("Fit Latency", f"{m_res['fit_duration_sec']}s")
            primary_score = m_res["metrics"].get("macro_f1" if m_res["task_type"] == "classification" else "r2", "N/A")
            m_col4.metric("Primary Test Metric", f"{primary_score}")

            st.json(m_res["metrics"])

        # Case B: Recommender Query Results (Defect C6)
        elif inspected_step.op_type == "recommender_query" and inspected_step.output_model_result:
            st.markdown("#### 🎬 Recommender Results Scorecard")
            r_res = inspected_step.output_model_result
            st.write(f"**Query Item:** `{r_res['query_title']}` • **Vocabulary Size:** `{r_res['vocab_size']:,}` terms • **Latency:** `{r_res['query_duration_sec']}s`")
            if inspected_step.output_df is not None:
                st.dataframe(inspected_step.output_df, use_container_width=True)

        # Case C: Pure Data Transformation & What Changed? (Defect C6)
        elif inspected_step.what_changed:
            st.markdown("#### 📊 Transformation Delta Ledger")
            diff = inspected_step.what_changed

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Rows Before → After", f"{diff['rows_before']:,} → {diff['rows_after']:,}", delta=f"{diff['rows_delta']:,} rows")
            d2.metric("Columns Added", f"{len(diff['columns_added'])}", delta=", ".join(diff['columns_added'][:3]) if diff['columns_added'] else "None")
            d3.metric("Columns Removed", f"{len(diff['columns_removed'])}", delta=", ".join(diff['columns_removed'][:3]) if diff['columns_removed'] else "None")
            d4.metric("Missing Cells Delta", f"{diff['missing_cells_delta']:,}")

            if inspected_step.output_df is not None:
                st.markdown("##### 📄 Output DataFrame Preview (Top 15 Rows)")
                st.dataframe(inspected_step.output_df.head(15), use_container_width=True)

                csv_step = io.StringIO()
                inspected_step.output_df.to_csv(csv_step, index=False)
                st.download_button(
                    label=f"📥 Download Step #{inspected_step.name} Output (CSV)",
                    data=csv_step.getvalue(),
                    file_name=f"step_{inspected_step.step_id}_output.csv",
                    mime="text/csv"
                )
