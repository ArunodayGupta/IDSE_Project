"""
missions_ui.py
Phase 4: Data Scientist Mission Mode Interactive UI.

Provides:
- Guided interactive mission tracker with badge showcase and progress tracking.
- Layered hints (Conceptual -> Concrete Dataset Hint -> Step-by-step Walkthrough).
- Verifiable evidence checks connected directly to actions across DataLab.
- Interactive reflection questions teaching real-world data science trade-offs.
- Seamless dataset adaptation for Netflix catalog vs custom uploaded tabular files.
"""

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd

from .data_context import get_source_dataset, get_working_dataset
from .missions import list_all_missions, get_mission, DataScientistMission


def render_missions_tab(netflix_raw_df: Optional[pd.DataFrame] = None) -> None:
    """
    Renders the Data Scientist Mission Mode guided learning environment.
    """
    st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #6F42C1; border-radius: 8px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h2 style="margin: 0; font-size: 1.45rem; font-weight: 700; color: #2B3035;">
                        🎯 Data Scientist Mission Mode
                    </h2>
                    <p style="margin: 4px 0 0 0; color: #6C757D; font-size: 0.9rem;">
                        Learn core data science by executing practical, evidence-verified workflows against real data.
                    </p>
                </div>
                <span style="background: #F3EEFD; color: #6F42C1; border: 1px solid #D5C2F6; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                    GUIDED MISSION TRACK
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Retrieve current active dataset context
    source_df, source_ident, source_name = get_source_dataset()
    working_df, _, _, is_transformed, _ = get_working_dataset()
    active_df = working_df if (is_transformed and working_df is not None) else source_df
    if active_df is None and netflix_raw_df is not None:
        active_df = netflix_raw_df

    # Session State: Completed Missions & Progress
    completed_missions: Dict[str, Dict[str, Any]] = st.session_state.setdefault("datalab_completed_missions", {})
    all_missions = list_all_missions()
    total_count = len(all_missions)
    done_count = len(completed_missions)
    progress_pct = done_count / total_count if total_count > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 1: Progress Tracker & Badge Showcase
    # ------------------------------------------------------------------
    prog_c1, prog_c2, prog_c3 = st.columns([2, 3, 2])
    with prog_c1:
        st.metric("Missions Completed", f"{done_count} / {total_count}", delta=f"{int(progress_pct * 100)}% Progress")
    with prog_c2:
        st.write("")
        st.progress(progress_pct)
        st.caption(f"Curriculum Track: {done_count} of {total_count} data science missions completed.")
    with prog_c3:
        if st.button("🔄 Reset Mission Progress", key="btn_reset_all_missions"):
            st.session_state["datalab_completed_missions"] = {}
            st.session_state["mission_reflection_answers"] = {}
            st.success("Mission progress reset!")
            st.rerun()

    # Badges Row
    st.markdown("##### 🏅 Earned Data Scientist Badges")
    badge_cols = st.columns(len(all_missions))
    for idx, m in enumerate(all_missions):
        with badge_cols[idx]:
            is_done = m.mission_id in completed_missions
            badge_bg = "#E6FBF2" if is_done else "#F8F9FA"
            badge_border = "#00D97E" if is_done else "#DEE2E6"
            badge_color = "#008A4F" if is_done else "#ADB5BD"
            check_mark = "✅ " if is_done else "🔒 "

            st.markdown(f"""
                <div style="background-color: {badge_bg}; border: 1px solid {badge_border}; border-radius: 6px; padding: 10px 6px; text-align: center;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: {badge_color}; line-height: 1.2;">
                        {check_mark}{m.badge}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Step 2: Mission Selector & Stage Details
    # ------------------------------------------------------------------
    sel_col, detail_col = st.columns([1, 2])

    with sel_col:
        st.markdown("##### 🗺️ Mission Track")
        mission_keys = [m.mission_id for m in all_missions]

        # Determine default active mission
        default_idx = 0
        for i, m in enumerate(all_missions):
            if m.mission_id not in completed_missions:
                default_idx = i
                break

        selected_m_id = st.radio(
            "Select Mission:",
            options=mission_keys,
            index=default_idx,
            format_func=lambda x: f"{'✅' if x in completed_missions else '⏳'} {get_mission(x).title.split(':')[1].strip()}",
            key="rb_selected_mission"
        )

    current_mission = get_mission(selected_m_id)
    is_mission_completed = (current_mission.mission_id in completed_missions)

    with detail_col:
        st.markdown(f"### {current_mission.title}")
        diff_color = "#00D97E" if current_mission.difficulty == "Beginner" else ("#2C7BE5" if current_mission.difficulty == "Intermediate" else "#6F42C1")
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="background: #F8F9FA; color: {diff_color}; border: 1px solid {diff_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                    {current_mission.difficulty.upper()}
                </span>
                <span style="font-size: 0.85rem; color: #6C757D;">
                    Award Badge: <b>{current_mission.badge}</b>
                </span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**🎯 Learning Objective:** {current_mission.objective}")

        # Check dataset prerequisites (Defect D4)
        is_compat, compat_msg = current_mission.is_compatible(active_df)
        if not is_compat:
            st.warning(f"⚠️ {compat_msg}")

        st.markdown("##### 📋 Mission Steps")
        for s in current_mission.steps:
            st.markdown(f"- {s}")

        st.markdown("---")

        # Layered Hints Accordion (Defect D5)
        st.markdown("##### 💡 Guided Hints")
        with st.expander("Level 1: Conceptual Principle"):
            st.write(current_mission.hints.get(1, "Think about the statistical trade-offs."))

        with st.expander("Level 2: Dataset Context"):
            st.write(current_mission.hints.get(2, "Review the active columns in your current dataset."))

        with st.expander("Level 3: Step-by-Step Walkthrough"):
            st.write(current_mission.hints.get(3, "Follow the exact button sequence in DataLab."))

        st.markdown("---")

        # --------------------------------------------------------------
        # Step 3: Verifiable Action Verification & Evidence (Defect D1)
        # --------------------------------------------------------------
        st.markdown("##### 🔬 Verification & Evidence")
        verify_btn = st.button("🔍 Verify Mission Completion", type="primary", use_container_width=True)

        if verify_btn:
            # First mark user explored if on mission 1
            if current_mission.mission_id == "mission_1":
                st.session_state["datalab_user_explored"] = True

            ok, message, evidence = current_mission.verification_fn(active_df or pd.DataFrame(), dict(st.session_state))

            if ok:
                completed_missions[current_mission.mission_id] = {
                    "completed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "evidence": evidence,
                }
                st.session_state["datalab_completed_missions"] = completed_missions
                st.success(f"🎉 **Mission Accomplished!** {message}")
                st.balloons()
            else:
                st.error(f"❌ **Prerequisites Not Yet Met:** {message}")

        # If completed, show Reflection Question & Trade-off Explanation (Defect D6)
        if current_mission.mission_id in completed_missions:
            evidence_rec = completed_missions[current_mission.mission_id].get("evidence", {})
            if evidence_rec:
                st.markdown("##### 📄 Recorded Completion Evidence")
                st.json(evidence_rec)

            st.markdown("##### 🤔 Critical Data Science Reflection")
            q_data = current_mission.reflection_question
            q_key = f"refl_{current_mission.mission_id}"
            user_choice = st.radio(
                q_data["question"],
                options=q_data["options"],
                key=q_key
            )

            chosen_idx = q_data["options"].index(user_choice) if user_choice in q_data["options"] else -1
            if chosen_idx == q_data["correct_idx"]:
                st.success(f"✅ **Correct!** {q_data['explanation']}")
            else:
                st.info(f"💡 **Analysis:** {q_data['explanation']}")

            st.info(f"📚 **Key Takeaway:** {current_mission.tradeoff_explanation}")

            if current_mission.next_mission_id:
                next_m = get_mission(current_mission.next_mission_id)
                st.markdown(f"👉 **Next Up:** *{next_m.title}*")
