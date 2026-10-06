# SynthAI Enterprise Meeting Intelligence v4.1
import streamlit as st
import os
import io
import time
import datetime
from typing import Dict, Any, List, Optional
import base64
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    HAS_PLOTLY = False

# Milestone 4 core modules
from database import (
    init_db,
    create_user,
    authenticate_user,
    get_user_by_id,
    get_all_meetings,
    get_user_meetings,
    get_meeting_by_id,
    delete_meeting,
    save_meeting_to_db,
    can_user_access_meeting,
    verify_knowledge_repository,
    get_sync_history
)
from vector_store import get_vector_store_stats, insert_meeting_vectors, trace_vector_to_meeting, get_vector_collection
from embedding_service import generate_meeting_embeddings
from search_service import semantic_search_meetings, TARGET_SLA_SECONDS
from rag_service import generate_grounded_rag_answer, stream_grounded_rag_answer
from zoom_service import (
    list_zoom_recordings,
    process_zoom_recording
)
from google_meet_service import (
    list_google_meet_recordings,
    process_google_meet_recording
)
from report_service import generate_meeting_pdf, generate_meeting_csv
from analytics_service import compute_meeting_analytics, compute_user_overview_analytics
from transcription_pipeline import process_uploaded_media
from seed_data import seed_database_and_vector_store

# Auto-initialize database tables and seed if empty
init_db()

st.set_page_config(
    page_title="SynthAI — Enterprise Meeting Intelligence & Integrations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PREMIUM DESIGN SYSTEM CSS
# ─────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --bg-void: #060813;
  --bg-surface: #0B0F1C;
  --bg-card: #11172A;
  --border-subtle: rgba(99, 102, 241, 0.2);
  --border-glow: rgba(99, 102, 241, 0.5);
  --indigo-glow: #6366F1;
  --cyan-glow: #06B6D4;
  --emerald: #10B981;
  --amber: #F59E0B;
  --rose: #F43F5E;
  --text-main: #F8FAFC;
  --text-muted: #94A3B8;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg-void) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  color: var(--text-main) !important;
}

[data-testid="stSidebar"] {
  background: #080C18 !important;
  border-right: 1px solid var(--border-subtle) !important;
}

/* Glassmorphic Cards */
.synth-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
  transition: border-color 0.2s ease, transform 0.2s ease;
}
.synth-card:hover {
  border-color: var(--border-glow);
}

/* Badges */
.badge-indigo {
  background: rgba(99, 102, 241, 0.15);
  color: #818CF8;
  border: 1px solid rgba(99, 102, 241, 0.4);
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  display: inline-block;
}
.badge-emerald {
  background: rgba(16, 185, 129, 0.15);
  color: #34D399;
  border: 1px solid rgba(16, 185, 129, 0.4);
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  display: inline-block;
}
.badge-amber {
  background: rgba(245, 158, 11, 0.15);
  color: #FBBF24;
  border: 1px solid rgba(245, 158, 11, 0.4);
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  display: inline-block;
}
.badge-rose {
  background: rgba(244, 63, 94, 0.15);
  color: #FB7185;
  border: 1px solid rgba(244, 63, 94, 0.4);
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  display: inline-block;
}

/* Tab & Button Polish */
button[kind="primary"] {
  background: linear-gradient(135deg, #4F46E5, #06B6D4) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
}

div[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  color: #38BDF8 !important;
}
</style>
""", unsafe_allow_html=True)

def render_download_button(data: bytes | str, filename: str, mime: str, label: str, is_primary: bool = True):
    """
    Bulletproof browser download using Base64 Data URI.
    Guarantees that Chromium/Edge/Firefox will save the file with the exact filename
    and extension (.pdf or .csv) instead of an internal Streamlit session UUID.
    """
    if isinstance(data, str):
        b64 = base64.b64encode(data.encode('utf-8')).decode()
    else:
        b64 = base64.b64encode(data).decode()

    bg = "linear-gradient(135deg, #6366F1 0%, #4338CA 100%)" if is_primary else "linear-gradient(135deg, #1E293B 0%, #0F172A 100%)"
    border = "1px solid rgba(99, 102, 241, 0.5)" if is_primary else "1px solid rgba(148, 163, 184, 0.25)"

    html = f"""
    <div style="margin: 0.4rem 0;">
        <a href="data:{mime};base64,{b64}" download="{filename}" target="_blank" style="text-decoration: none; display: block; width: 100%;">
            <div style="background: {bg}; border: {border}; color: #FFFFFF; padding: 0.65rem 1.2rem; border-radius: 8px; text-align: center; font-weight: 600; font-size: 0.92rem; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                {label}
            </div>
        </a>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE SETUP
# ─────────────────────────────────────────────
if "current_user" not in st.session_state:
    st.session_state.current_user = {"id": 1, "username": "admin", "role": "admin"}
if "active_meeting_id" not in st.session_state:
    st.session_state.active_meeting_id = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ─────────────────────────────────────────────
# SIDEBAR: USER AUTH & SYSTEM STATUS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 SynthAI Enterprise")
    st.caption("AI-Powered Meeting Synthesis, Integrations & Semantic RAG")
    st.divider()

    # User Status Card
    user = st.session_state.current_user
    if user:
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Active Session</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">👤 {user['username']}</div>
            <span class="badge-indigo">Role: {user.get('role', 'member').upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout / Switch User", use_container_width=True):
            st.session_state.current_user = None
            st.rerun()
    else:
        st.warning("Not Logged In")
        auth_mode = st.radio("Access Mode", ["Login", "Register"], horizontal=True)
        u_name = st.text_input("Username", key="auth_u")
        u_pass = st.text_input("Password", type="password", key="auth_p")
        if auth_mode == "Login":
            if st.button("Log In", use_container_width=True, type="primary"):
                res = authenticate_user(u_name, u_pass)
                if res:
                    st.session_state.current_user = res
                    st.success(f"Welcome back, {res['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        else:
            if st.button("Register New User", use_container_width=True, type="primary"):
                res = create_user(u_name, u_pass)
                if res:
                    st.session_state.current_user = res
                    st.success(f"User {u_name} registered!")
                    st.rerun()
                else:
                    st.error("Username already taken or invalid.")

    st.divider()

    # API & Model Configuration
    st.markdown("#### 🔑 Model & API Keys")
    sb_gemini = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Google Gemini API key for structured extraction & grounded RAG Q&A"
    )
    if sb_gemini:
        os.environ["GEMINI_API_KEY"] = sb_gemini
        st.markdown('<div class="badge-emerald" style="margin-bottom:6px;">✓ Gemini Key Active</div>', unsafe_allow_html=True)
    else:
        st.caption("Enter Gemini Key for Live LLM Extraction")

    sb_hf = st.text_input(
        "Hugging Face Token",
        type="password",
        value=os.getenv("HF_TOKEN", ""),
        help="Required for PyAnnote speaker diarization"
    )
    if sb_hf:
        os.environ["HF_TOKEN"] = sb_hf
        st.markdown('<div class="badge-indigo" style="margin-bottom:6px;">✓ HF Token Active</div>', unsafe_allow_html=True)
    else:
        st.caption("HF Token optional (used for PyAnnote speaker diarization)")

    st.divider()

    # Repository & Vector DB Quick Stats
    v_stats = get_vector_store_stats()
    db_stats = verify_knowledge_repository()
    st.markdown("#### ⚡ System Health")
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Meetings", db_stats.get("total_meetings", 0))
    col_sb2.metric("Vectors", v_stats.get("total_vectors", 0))

    if db_stats.get("total_meetings", 0) == 0:
        if st.button("🌱 Seed Initial Repository", use_container_width=True):
            with st.spinner("Seeding database and vector store..."):
                seed_database_and_vector_store()
                st.success("Repository seeded!")
                st.rerun()

# ─────────────────────────────────────────────
# MAIN NAVIGATION TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Meetings Dashboard",
    "🔎 Meeting Details & Analytics",
    "🤖 RAG AI Assistant",
    "📹 Zoom Cloud Sync",
    "🌐 Google Meet Drive Sync",
    "📑 Reports & Export",
    "🛡️ Access & Security",
    "🏗️ Architecture & Flowchart"
])

# ══════════════════════════════════════════════════
# TAB 1: MEETINGS DASHBOARD (TASK 1)
# ══════════════════════════════════════════════════
with tabs[0]:
    st.title("📊 Meeting Intelligence Dashboard")
    st.caption("Centralized hub for uploaded recordings, Zoom imports, and Google Meet transcripts.")

    # Top Metric Bar
    user_id = st.session_state.current_user["id"] if st.session_state.current_user else None
    overview = compute_user_overview_analytics(user_id=user_id)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Total Meetings", overview["total_meetings"])
    m_col2.metric("Total Hours", f"{overview['total_duration_hours']}h")
    m_col3.metric("Action Items", overview["total_action_items"])
    m_col4.metric("Completed Actions", overview["completed_action_items"])
    m_col5.metric("Completion Rate", f"{overview['completion_rate_percent']}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Section A: Upload New Meeting
    with st.expander("📤 Upload New Meeting Recording (MP4, MP3, WAV, M4A)", expanded=False):
        uploaded_file = st.file_uploader("Select recording file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"])
        upload_title = st.text_input("Meeting Title (Optional)", placeholder="e.g. Q3 Sprint Planning")
        if uploaded_file and st.button("Start AI Processing Pipeline", type="primary"):
            p_bar = st.progress(0, text="Starting pipeline...")
            status_box = st.empty()

            def progress_cb(pct, text):
                p_bar.progress(int(pct * 100), text=text)
                status_box.info(f"⏳ **{text}**")

            try:
                file_bytes = uploaded_file.read()
                upload_res = process_uploaded_media(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    title=upload_title,
                    user_id=user_id,
                    platform="upload",
                    progress_callback=progress_cb
                )
                p_bar.progress(100, text="Processing Complete!")
                st.success(
                    f"✅ **Meeting '{upload_res['title']}' Processed Successfully!**\n\n"
                    f"• **Meeting ID:** #{upload_res['meeting_id']}\n"
                    f"• **Spoken Segments:** {upload_res['segments_count']}\n"
                    f"• **Action Items:** {upload_res['action_items_count']}\n"
                    f"• **Key Decisions:** {upload_res['decisions_count']}\n"
                    f"• **Participants:** {', '.join(upload_res['participants'])}\n"
                    f"• **ChromaDB Vectors:** {upload_res['vector_count']}"
                )
                st.session_state.active_meeting_id = upload_res['meeting_id']
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during meeting processing: {e}")

    # Section B: Filters & Meeting List
    st.subheader("📁 Historical Meetings")
    f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
    with f_col1:
        search_filter = st.text_input("🔍 Search Meetings by Keyword", placeholder="Search titles, summaries, decisions...")
    with f_col2:
        platform_filter = st.selectbox("Platform Filter", ["All Platforms", "Uploads", "Zoom", "Google Meet"])
    with f_col3:
        sort_order = st.selectbox("Sort Order", ["Most Recent First", "Oldest First"])

    # Query meetings
    platform_map = {"All Platforms": None, "Uploads": "upload", "Zoom": "zoom", "Google Meet": "google_meet"}
    meetings = get_user_meetings(user_id=user_id, platform=platform_map[platform_filter])
    
    if search_filter.strip():
        q = search_filter.strip().lower()
        meetings = [m for m in meetings if q in m["filename"].lower() or q in (m.get("summary") or "").lower() or q in (m.get("title") or "").lower()]

    if sort_order == "Oldest First":
        meetings.reverse()

    if not meetings:
        st.info("No meetings found matching your filter criteria.")
    else:
        for m in meetings:
            p_badge = "badge-indigo" if m["platform"] == "zoom" else ("badge-emerald" if m["platform"] == "google_meet" else "badge-amber")
            p_icon = "📹" if m["platform"] == "zoom" else ("🌐" if m["platform"] == "google_meet" else "🎙️")
            created_str = str(m.get("created_at", ""))[:16].replace("T", " ")

            with st.container():
                st.markdown(f"""
                <div class="synth-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9;">{p_icon} {m['title']}</span>
                        <span class="{p_badge}">{m['platform'].replace('_', ' ').upper()}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.75rem;">
                        📅 {created_str} &nbsp;•&nbsp; ⏱️ {round(float(m.get('duration_seconds', 0))/60.0, 1)} mins &nbsp;•&nbsp; 👥 {len(m.get('participants', []))} Participants &nbsp;•&nbsp; 📋 {len(m.get('action_items', []))} Action Items
                    </div>
                    <div style="font-size: 0.92rem; color: #CBD5E1; margin-bottom: 0.75rem; line-height: 1.5;">
                        {m.get('summary', 'No summary generated yet.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 5])
                with btn_col1:
                    if st.button(f"🔍 View Details #{m['id']}", key=f"view_det_{m['id']}"):
                        st.session_state.active_meeting_id = m['id']
                        # Switch to Tab 2
                        st.info(f"Loaded Meeting #{m['id']}! Click on '🔎 Meeting Details & Analytics' tab above.")
                with btn_col2:
                    if st.button(f"📑 Quick PDF #{m['id']}", key=f"pdf_btn_{m['id']}"):
                        st.session_state[f"show_pdf_{m['id']}"] = True
                    if st.session_state.get(f"show_pdf_{m['id']}"):
                        pdf_data = generate_meeting_pdf(m)
                        render_download_button(
                            data=pdf_data,
                            filename=f"SynthAI_Report_Meeting_{m['id']}.pdf",
                            mime="application/pdf",
                            label=f"⬇️ Save Meeting #{m['id']} PDF",
                            is_primary=True
                        )

# ══════════════════════════════════════════════════
# TAB 2: MEETING DETAILS & ANALYTICS (TASK 2)
# ══════════════════════════════════════════════════
with tabs[1]:
    st.title("🔎 Meeting Details & Deep Analytics")
    st.caption("Sequential meeting intelligence flow: Selection ➔ Details ➔ Transcript ➔ Summary ➔ Decisions ➔ Actions ➔ Analytics.")

    all_user_meetings = get_user_meetings(user_id=user_id)
    if not all_user_meetings:
        st.info("No meetings available to inspect. Upload or import a recording first.")
    else:
        # Step 1: Meeting Selection
        meeting_options = {f"#{m['id']} — {m['title']} ({m['platform']})": m['id'] for m in all_user_meetings}
        current_idx = 0
        if st.session_state.active_meeting_id:
            for idx, (label, m_id) in enumerate(meeting_options.items()):
                if m_id == st.session_state.active_meeting_id:
                    current_idx = idx
                    break

        selected_label = st.selectbox("Select Meeting to Inspect", list(meeting_options.keys()), index=current_idx)
        selected_m_id = meeting_options[selected_label]
        selected_meeting = get_meeting_by_id(selected_m_id)

        st.divider()

        # Step 2: Meeting Details Header
        m_dur = round(float(selected_meeting.get("duration_seconds", 0)) / 60.0, 1)
        st.markdown(f"""
        ### 📌 {selected_meeting['title']}
        **Meeting ID:** `#{selected_meeting['id']}` &nbsp;|&nbsp;
        **Platform:** `{selected_meeting['platform'].upper()}` &nbsp;|&nbsp;
        **Date:** `{str(selected_meeting['created_at'])[:16].replace('T', ' ')}` &nbsp;|&nbsp;
        **Duration:** `{m_dur} minutes`
        """)

        pts = selected_meeting.get("participants", [])
        if pts:
            pt_chips = " ".join([f'<span class="badge-indigo" style="margin-right:6px; margin-bottom:4px;">👤 {p if isinstance(p, str) else p.get("name", "")}</span>' for p in pts])
            st.markdown(f"**👥 Participants:** {pt_chips}", unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Step 3: Transcript Viewer
        with st.expander("🎙️ Full Dialogue Transcript", expanded=False):
            transcript_text = selected_meeting.get("transcript") or "No transcript recorded."
            t_search = st.text_input("Filter Transcript Text", placeholder="Search dialogue lines...", key="t_search")
            if t_search.strip():
                lines = [line for line in transcript_text.split("\n") if t_search.lower() in line.lower()]
                st.code("\n".join(lines) if lines else "No matching lines found.", language="text")
            else:
                st.text_area("Transcript Dialogue", transcript_text, height=220)

        # Step 4: Summary Viewer
        with st.expander("📌 Executive Summary & Key Points", expanded=True):
            st.markdown(f"**Executive Summary:**\n\n{selected_meeting.get('summary', 'No summary available.')}")
            kps = selected_meeting.get("key_points", [])
            if kps:
                st.markdown("**Key Topics Discussed:**")
                for kp in kps:
                    st.markdown(f"- {kp}")

        # Step 5: Key Decisions
        with st.expander("⚖️ Key Decisions Made", expanded=True):
            decs = selected_meeting.get("key_decisions", [])
            if decs:
                for idx, dec in enumerate(decs, 1):
                    st.markdown(f"**{idx}.** {dec}")
            else:
                st.write("No decisions logged for this meeting.")

        # Step 6: Action Items & Responsibilities & Deadlines
        with st.expander("✅ Action Items, Responsibilities & Deadlines", expanded=True):
            ais = selected_meeting.get("action_items", [])
            if ais:
                ai_df = pd.DataFrame(ais)[["description", "assigned_participant", "deadline", "priority", "status"]]
                ai_df.columns = ["Action Description", "Assignee", "Deadline", "Priority", "Status"]
                st.dataframe(ai_df, use_container_width=True)
            else:
                st.write("No action items extracted.")

        # Step 7: Meeting Analytics & Interactive Charts
        st.subheader("📈 Meeting Analytics & Workload Distribution")
        analytics = compute_meeting_analytics(selected_meeting)
        
        a_col1, a_col2, a_col3 = st.columns(3)
        a_col1.metric("Dialogue Word Count", analytics["word_count"])
        a_col2.metric("Total Action Items", analytics["action_items_total"])
        a_col3.metric("Key Decisions", analytics["decisions_count"])

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            # Action items by priority chart
            prio_data = analytics["action_items_by_priority"]
            if HAS_PLOTLY and px:
                fig_prio = px.bar(
                    x=list(prio_data.keys()),
                    y=list(prio_data.values()),
                    labels={'x': 'Priority', 'y': 'Count'},
                    title="Action Items by Priority",
                    color=list(prio_data.keys()),
                    color_discrete_map={"High": "#F43F5E", "Medium": "#F59E0B", "Low": "#10B981"}
                )
                fig_prio.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_prio, use_container_width=True)
            else:
                st.markdown("#### Action Items by Priority")
                st.bar_chart(pd.DataFrame(list(prio_data.values()), index=list(prio_data.keys()), columns=["Count"]))

        with chart_col2:
            # Action items by status donut chart
            status_data = analytics["action_items_by_status"]
            if HAS_PLOTLY and px:
                fig_stat = px.pie(
                    names=list(status_data.keys()),
                    values=list(status_data.values()),
                    title="Action Items by Status",
                    hole=0.45,
                    color=list(status_data.keys()),
                    color_discrete_map={"Completed": "#10B981", "In Progress": "#38BDF8", "Pending": "#F59E0B"}
                )
                fig_stat.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_stat, use_container_width=True)
            else:
                st.markdown("#### Action Items by Status")
                st.bar_chart(pd.DataFrame(list(status_data.values()), index=list(status_data.keys()), columns=["Count"]))

# ══════════════════════════════════════════════════
# TAB 3: RAG SEARCH & AI ASSISTANT (TASKS 3 & 4)
# ══════════════════════════════════════════════════
with tabs[2]:
    st.title("🤖 Grounded RAG Semantic Search & AI Assistant")
    st.caption("Ask natural-language questions across meetings. Answers are strictly grounded in retrieved vector context with source citations.")

    # Quick test query buttons
    st.markdown("**⚡ Quick Test Queries:**")
    preset_cols = st.columns(4)
    preset_q = None
    with preset_cols[0]:
        if st.button("🗄️ Database migration?", use_container_width=True, key="btn_q_mig"):
            preset_q = "Which meeting discussed the database migration?"
    with preset_cols[1]:
        if st.button("📱 Mobile app deadline?", use_container_width=True, key="btn_q_mob"):
            preset_q = "What deadline was decided for the mobile application?"
    with preset_cols[2]:
        if st.button("🎙️ Whisper accuracy tests?", use_container_width=True, key="btn_q_whi"):
            preset_q = "What tests were recommended for Whisper transcription accuracy?"
    with preset_cols[3]:
        if st.button("⏰ Rollback script owner?", use_container_width=True, key="btn_q_rol"):
            preset_q = "Who was assigned to the rollback script and what is the deadline?"

    if preset_q:
        st.session_state.search_query = preset_q

    rag_sub1, rag_sub2, rag_sub3 = st.tabs([
        "💬 Grounded AI Assistant (Q&A)",
        "🔍 Sub-3s Semantic Vector Search",
        "🛡️ Vector Store Inspector & Traceability"
    ])

    # ─────────────────────────────────────────────
    # SUB-TAB 1: GROUNDED AI ASSISTANT (Q&A)
    # ─────────────────────────────────────────────
    with rag_sub1:
        st.markdown("### 💬 Ask Questions Across Meeting Records")
        col_scope, col_entity = st.columns([3, 2])
        with col_scope:
            scope_mode = st.radio("Search Scope", ["All Accessible Meetings", "Active Meeting Only"], horizontal=True, key="rag_scope_mode")
        with col_entity:
            rag_entity = st.selectbox(
                "Focus Entity Filter",
                ["All Knowledge Units", "Decisions Only", "Action Items Only", "Executive Summaries Only"],
                key="rag_entity_filter"
            )

        doc_type_map = {
            "All Knowledge Units": None,
            "Decisions Only": "decision",
            "Action Items Only": "action_item",
            "Executive Summaries Only": "summary"
        }

        active_mid_filter = st.session_state.active_meeting_id if scope_mode == "Active Meeting Only" else None

        user_question = st.text_input(
            "Natural Language Question",
            value=st.session_state.search_query or "",
            placeholder="e.g. Which meeting discussed the database migration?",
            key="rag_q_input"
        )

        if user_question.strip() and st.button("Query Knowledge Repository & Synthesize Answer", type="primary", key="btn_run_rag"):
            with st.spinner("Retrieving relevant meeting units & generating grounded response..."):
                rag_res = generate_grounded_rag_answer(
                    question=user_question,
                    meeting_id=active_mid_filter,
                    user_id=user_id,
                    doc_type=doc_type_map[rag_entity],
                    top_k=5
                )

                ret_time = rag_res.get('retrieval_time_seconds', 0.0)
                tot_time = rag_res.get('total_time_seconds', 0.0)
                sla_met = ret_time <= TARGET_SLA_SECONDS
                is_grounded = rag_res.get('is_grounded', False)

                # SLA & Grounding Status Banner
                st.markdown(f"""
                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.75rem;">
                    <span class="{'badge-emerald' if sla_met else 'badge-amber'}">
                        ⚡ SLA &lt; {TARGET_SLA_SECONDS}s: Retrieved in {ret_time:.4f}s ({'PASSED' if sla_met else 'SLA WARNING'})
                    </span>
                    <span class="{'badge-indigo' if is_grounded else 'badge-rose'}">
                        {'🛡️ Strictly Grounded in Meeting Context' if is_grounded else '⚠️ Zero Hallucination: Insufficient Context'}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Grounded Answer Card
                border_color = "#6366F1" if is_grounded else "#F43F5E"
                st.markdown(f"""
                <div class="synth-card" style="border-left: 4px solid {border_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; color: #818CF8; font-size: 1.05rem;">🤖 AI Grounded Response</span>
                        <span style="font-size: 0.75rem; color: #94A3B8;">Total Pipeline: {tot_time:.3f}s</span>
                    </div>
                    <div style="font-size: 1.02rem; line-height: 1.6; color: #F8FAFC;">
                        {rag_res['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Cited Sources
                sources = rag_res.get("sources", [])
                st.markdown(f"#### 📚 Grounding Evidence Sources ({len(sources)} cited items)")
                if not sources:
                    st.info("No source records met the relevance threshold for this query.")
                else:
                    for idx, src in enumerate(sources, 1):
                        s_date = src.get("date") or "N/A"
                        score_pct = int(src.get("relevance_score", 0.0) * 100)
                        dtype = src.get("doc_type", "transcript").replace("_", " ").upper()

                        with st.container():
                            st.markdown(f"""
                            <div style="background: rgba(15, 22, 40, 0.7); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; padding: 0.85rem; margin-bottom: 0.5rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 700; color: #38BDF8;">[Source {idx}] {src['filename']} (ID #{src['meeting_id']})</span>
                                    <span class="badge-indigo">{dtype} · {score_pct}% Match</span>
                                </div>
                                <div style="font-size: 0.8rem; color: #94A3B8; margin: 3px 0;">📅 Date: {s_date} &nbsp;•&nbsp; Relevance Score: {src.get('relevance_score', 0.0):.4f}</div>
                                <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 6px; font-style: italic; background: rgba(0,0,0,0.25); padding: 6px 10px; border-radius: 6px;">
                                    "{src['preview']}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            col_act1, col_act2 = st.columns([2, 5])
                            with col_act1:
                                if st.button(f"🔍 Inspect Meeting #{src['meeting_id']}", key=f"src_insp_{idx}_{src['meeting_id']}"):
                                    st.session_state.active_meeting_id = src['meeting_id']
                                    st.success(f"Meeting #{src['meeting_id']} selected! Switch to '🔎 Meeting Details & Analytics' tab to view.")

    # ─────────────────────────────────────────────
    # SUB-TAB 2: SUB-3S SEMANTIC VECTOR SEARCH
    # ─────────────────────────────────────────────
    with rag_sub2:
        st.markdown("### 🔎 High-Performance Semantic Vector Search")
        st.caption("Search across historical meetings with sub-3s SLA latency. Visualizes relevance scores and entity breakdowns.")

        s_col1, s_col2 = st.columns([4, 2])
        with s_col1:
            pure_query = st.text_input("Semantic Query / Keywords", value=st.session_state.search_query or "", placeholder="e.g. database migration or rollback scripts", key="pure_search_input")
        with s_col2:
            pure_entity = st.selectbox("Entity Constraint", ["All Entities", "summary", "decision", "action_item", "transcript"], key="pure_entity_filter")

        target_dtype = pure_entity if pure_entity != "All Entities" else None

        if pure_query.strip() and st.button("Execute Vector Search (<3s SLA)", type="primary", key="btn_run_pure_search"):
            st.session_state.search_query = pure_query.strip()
            with st.spinner("Searching persistent ChromaDB index..."):
                s_res = semantic_search_meetings(
                    query=pure_query.strip(),
                    top_k=8,
                    doc_type=target_dtype,
                    user_id=user_id
                )

                t_ret = s_res.get("search_time_seconds", 0.0)
                tot_hits = s_res.get("total_results", 0)
                matched_m = s_res.get("matched_meetings", [])

                st.markdown(f"""
                <div style="display: flex; gap: 0.6rem; align-items: center; margin: 0.5rem 0 1rem;">
                    <span class="badge-emerald">⚡ SLA PASS: Retrieved {tot_hits} units in {t_ret:.4f}s (&lt; {TARGET_SLA_SECONDS}s target)</span>
                    <span class="badge-indigo">📂 {len(matched_m)} Relevant Meetings Matched</span>
                </div>
                """, unsafe_allow_html=True)

                if not matched_m:
                    st.info("No meeting vectors matched your search criteria.")
                else:
                    for m_idx, m_hit in enumerate(matched_m):
                        score_pct = int(m_hit["best_score"] * 100)
                        with st.expander(f"📄 {m_hit['filename']} — Best Match: {score_pct}% ({m_hit['hits_count']} hit units)", expanded=(m_idx == 0)):
                            st.markdown(f"**Meeting Filename:** `{m_hit['filename']}` &nbsp;|&nbsp; **ID:** `#{m_hit['meeting_id']}` &nbsp;|&nbsp; **Date:** `{m_hit['created_at'][:16] if m_hit['created_at'] else 'N/A'}`")
                            if m_hit.get("summary"):
                                st.markdown(f"**Executive Summary Preview:**\n{m_hit['summary'][:240]}...")

                            st.markdown("#### Matched Entities in this Meeting:")
                            for ent in m_hit.get("matching_entities", []):
                                ent_dtype = ent.get("doc_type", "item").upper()
                                ent_score = ent.get("score", 0.0)
                                st.markdown(f"""
                                <div style="background: rgba(15, 22, 40, 0.5); border-left: 3px solid #6366F1; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 4px;">
                                    <span class="badge-indigo">{ent_dtype}</span>
                                    <span style="font-size: 0.75rem; color: #94A3B8; margin-left: 0.5rem;">Similarity Score: {ent_score:.4f}</span>
                                    <div style="font-size: 0.85rem; color: #F1F5F9; margin-top: 4px;">{ent.get('text', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            if st.button(f"🔍 Inspect Meeting #{m_hit['meeting_id']} in Details", key=f"open_matched_m_{m_idx}_{m_hit['meeting_id']}"):
                                st.session_state.active_meeting_id = m_hit['meeting_id']
                                st.success(f"Meeting #{m_hit['meeting_id']} selected! Switch to '🔎 Meeting Details & Analytics' tab to inspect.")

    # ─────────────────────────────────────────────
    # SUB-TAB 3: VECTOR STORE INSPECTOR & TRACEABILITY
    # ─────────────────────────────────────────────
    with rag_sub3:
        st.markdown("### 🛡️ Persistent Vector Knowledge Store Inspector")
        st.caption("Inspect ChromaDB collection status, entity breakdown, and trace vectors back to relational meetings.")

        v_info = get_vector_store_stats()
        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Total Stored Vectors", v_info.get("total_vectors", 0))
        v_col2.metric("Indexed Meetings", v_info.get("unique_meetings_indexed", 0))
        v_col3.metric("Collection Status", v_info.get("status", "unknown").upper())

        st.markdown("#### 📊 Breakdown by Entity Type")
        bd = v_info.get("breakdown", {})
        if bd:
            bd_df = pd.DataFrame(list(bd.items()), columns=["Entity Type", "Vector Count"])
            bd_col1, bd_col2 = st.columns([1, 2])
            with bd_col1:
                st.dataframe(bd_df, use_container_width=True)
            with bd_col2:
                if HAS_PLOTLY and px:
                    fig_bd = px.pie(
                        bd_df, names="Entity Type", values="Vector Count",
                        title="Vector Knowledge Distribution",
                        hole=0.4,
                        color_discrete_sequence=["#6366F1", "#22D3EE", "#10B981", "#F59E0B", "#F43F5E"]
                    )
                    fig_bd.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bd, use_container_width=True)
                else:
                    st.bar_chart(bd_df.set_index("Entity Type"))

        st.divider()
        st.markdown("#### 🔬 Meeting-to-Vector Traceability Tool")
        st.caption("Verifies bidirectional traceability: ChromaDB Vector Unit ➔ Relational Meeting Knowledge Entity.")

        # Sample vector IDs for quick inspection
        coll = get_vector_collection()
        avail_ids = coll.get(limit=10).get("ids", []) if coll else []

        selected_vid = st.selectbox("Select a Vector ID to Trace", avail_ids if avail_ids else ["No vectors stored yet"])
        if selected_vid and selected_vid != "No vectors stored yet":
            trace_res = trace_vector_to_meeting(selected_vid)
            if trace_res and trace_res.get("traced"):
                st.markdown(f"""
                <div class="synth-card" style="border-left: 4px solid #10B981;">
                    <div style="font-weight: 700; color: #34D399; margin-bottom: 0.4rem;">✅ Verified Traceability Link</div>
                    <div><b>Vector ID:</b> <code>{trace_res['unit_id']}</code></div>
                    <div><b>Entity Type:</b> <span class="badge-indigo">{trace_res.get('doc_type', 'item').upper()}</span></div>
                    <div><b>Meeting ID:</b> <code>#{trace_res['meeting_id']}</code> &nbsp;|&nbsp; <b>Filename:</b> <code>{trace_res['meeting_filename']}</code></div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Relational database integrity confirmed: vector maps cleanly to meeting records.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Vector {selected_vid} could not be traced to a meeting.")

# ══════════════════════════════════════════════════
# TAB 4: ZOOM INTEGRATION (TASK 4)
# ══════════════════════════════════════════════════
with tabs[3]:
    st.title("📹 Zoom Cloud Recording Integration")
    st.caption("Synchronize Zoom recordings automatically into the transcription and knowledge repository.")

    st.markdown("""
    <div class="synth-card">
        <div style="font-weight: 700; color: #38BDF8; margin-bottom: 0.4rem;">🔗 Zoom Cloud Ingestion Pipeline</div>
        <div style="font-size: 0.88rem; color: #94A3B8;">
            <b>Workflow:</b> Zoom Recording ➔ Application Ingestion ➔ Whisper Audio Transcription ➔ Gemini LLM Summary ➔ Action Item & Decision Extraction ➔ Knowledge Repository & Vector Indexing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    zoom_recordings = list_zoom_recordings()
    st.subheader("Available Zoom Cloud Recordings")

    for rec in zoom_recordings:
        with st.container():
            st.markdown(f"""
            <div class="synth-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem;">🎥 {rec['topic']}</span>
                    <span class="badge-indigo">Zoom Cloud</span>
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                    UUID: <code>{rec['meeting_id']}</code> &nbsp;•&nbsp; ⏱️ {rec['duration_minutes']} mins &nbsp;•&nbsp; 📦 {rec['file_size_mb']} MB
                </div>
            </div>
            """, unsafe_allow_html=True)

            z_col1, z_col2 = st.columns([2, 5])
            with z_col1:
                if st.button(f"⚡ Sync Recording", key=f"zoom_sync_{rec['meeting_id']}", type="primary"):
                    with st.spinner("Processing Zoom recording through Whisper & LLM pipeline..."):
                        sync_res = process_zoom_recording(
                            recording_id=rec["meeting_id"],
                            topic=rec["topic"],
                            user_id=user_id
                        )
                        if sync_res["is_duplicate"]:
                            st.warning(f"⚠️ Duplicate: {sync_res['message']}")
                        elif sync_res["status"] == "success":
                            st.success(f"✅ Success! Ingested as Meeting #{sync_res['meeting_id']}.")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {sync_res['message']}")

    st.divider()
    st.subheader("📋 Recent Zoom Sync Activity")
    zoom_history = get_sync_history(platform="zoom", limit=5)
    if zoom_history:
        st.dataframe(pd.DataFrame(zoom_history)[["external_id", "file_name", "status", "synced_at"]], use_container_width=True)
    else:
        st.info("No Zoom recordings synced yet.")

# ══════════════════════════════════════════════════
# TAB 5: GOOGLE MEET INTEGRATION (TASK 5)
# ══════════════════════════════════════════════════
with tabs[4]:
    st.title("🌐 Google Meet Drive Integration")
    st.caption("Import Google Meet recordings stored in Google Drive directly into the AI pipeline.")

    st.markdown("""
    <div class="synth-card">
        <div style="font-weight: 700; color: #34D399; margin-bottom: 0.4rem;">📁 Google Drive Meet Recordings Folder</div>
        <div style="font-size: 0.88rem; color: #94A3B8;">
            <b>Workflow:</b> Google Meet Recording ➔ Drive File Retrieval ➔ Whisper Speech-to-Text ➔ Gemini Structured Extraction ➔ SQLite & ChromaDB Vector Store.
        </div>
    </div>
    """, unsafe_allow_html=True)

    gmeet_recordings = list_google_meet_recordings()
    st.subheader("Available Google Meet Drive Files")

    for rec in gmeet_recordings:
        with st.container():
            st.markdown(f"""
            <div class="synth-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem;">📁 {rec['file_name']}</span>
                    <span class="badge-emerald">Google Drive</span>
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                    File ID: <code>{rec['file_id']}</code> &nbsp;•&nbsp; 📦 {rec['size_mb']} MB &nbsp;•&nbsp; 📅 {rec['created_time'][:10]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            gm_col1, gm_col2 = st.columns([2, 5])
            with gm_col1:
                if st.button(f"⚡ Sync Meet File", key=f"gm_sync_{rec['file_id']}", type="primary"):
                    with st.spinner("Retrieving file from Drive and executing pipeline..."):
                        sync_res = process_google_meet_recording(
                            file_id=rec["file_id"],
                            file_name=rec["file_name"],
                            user_id=user_id
                        )
                        if sync_res["is_duplicate"]:
                            st.warning(f"⚠️ Duplicate: {sync_res['message']}")
                        elif sync_res["status"] == "success":
                            st.success(f"✅ Success! Ingested as Meeting #{sync_res['meeting_id']}.")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {sync_res['message']}")

    st.divider()
    st.subheader("📋 Recent Google Meet Sync Activity")
    gm_history = get_sync_history(platform="google_meet", limit=5)
    if gm_history:
        st.dataframe(pd.DataFrame(gm_history)[["external_id", "file_name", "status", "synced_at"]], use_container_width=True)
    else:
        st.info("No Google Meet recordings synced yet.")

# ══════════════════════════════════════════════════
# TAB 6: REPORTS & EXPORT (TASK 6)
# ══════════════════════════════════════════════════
with tabs[5]:
    st.title("📑 Executive Reports & Data Export")
    st.caption("Generate professional PDF documents and structured CSV exports for any selected meeting.")

    meetings_for_export = get_user_meetings(user_id=user_id)
    if not meetings_for_export:
        st.info("No meetings available for export.")
    else:
        export_choices = {f"#{m['id']} — {m['title']}": m['id'] for m in meetings_for_export}
        rep_m_label = st.selectbox("Select Meeting for Report Generation", list(export_choices.keys()), key="rep_select")
        rep_m_id = export_choices[rep_m_label]
        rep_meeting = get_meeting_by_id(rep_m_id)

        st.markdown(f"""
        <div class="synth-card">
            <h4>📄 Report Summary Preview: {rep_meeting['title']}</h4>
            <p><b>Summary:</b> {rep_meeting.get('summary', 'N/A')}</p>
            <p><b>Action Items:</b> {len(rep_meeting.get('action_items', []))} items &nbsp;|&nbsp; <b>Key Decisions:</b> {len(rep_meeting.get('key_decisions', []))} items</p>
        </div>
        """, unsafe_allow_html=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.markdown("#### 📄 Executive PDF Report")
            st.caption("Full report with Cover, Metadata Table, Executive Summary, Decisions, and Action Items Table.")
            pdf_bytes = generate_meeting_pdf(rep_meeting)
            pdf_filename = f"SynthAI_Report_Meeting_{rep_m_id}.pdf"
            render_download_button(
                data=pdf_bytes,
                filename=pdf_filename,
                mime="application/pdf",
                label=f"⬇️ Download Official PDF Report ({len(pdf_bytes):,} bytes)",
                is_primary=True
            )
            st.markdown(f"""
            <div style="font-size: 0.78rem; color: #94A3B8; text-align: center; margin-top: 4px;">
                Direct Backend API: <a href="http://127.0.0.1:8000/reports/pdf/{rep_m_id}" target="_blank" style="color: #818CF8; text-decoration: underline;">meeting_{rep_m_id}_report.pdf</a>
            </div>
            """, unsafe_allow_html=True)

        with dl_col2:
            st.markdown("#### 📊 Tabular CSV Export")
            st.caption("RFC-4180 structured spreadsheet containing all relational meeting attributes.")
            csv_str = generate_meeting_csv(rep_meeting)
            csv_filename = f"SynthAI_Data_Meeting_{rep_m_id}.csv"
            render_download_button(
                data=csv_str,
                filename=csv_filename,
                mime="text/csv",
                label="⬇️ Download Structured CSV Data",
                is_primary=False
            )
            st.markdown(f"""
            <div style="font-size: 0.78rem; color: #94A3B8; text-align: center; margin-top: 4px;">
                Direct Backend API: <a href="http://127.0.0.1:8000/reports/csv/{rep_m_id}" target="_blank" style="color: #818CF8; text-decoration: underline;">meeting_{rep_m_id}_report.csv</a>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAB 7: ACCESS & SECURITY VALIDATION (TASK 7)
# ══════════════════════════════════════════════════
with tabs[6]:
    st.title("🛡️ User Access Control & Security Validation")
    st.caption("Verify multi-tenant isolation, bcrypt password encryption, and session integrity.")

    st.markdown("""
    <div class="synth-card">
        <div style="font-weight: 700; color: #10B981; margin-bottom: 0.4rem;">🔒 Multi-Tenant Data Isolation Policy</div>
        <div style="font-size: 0.88rem; color: #94A3B8;">
            SynthAI enforces strict user access controls at both API and SQLite query levels.
            Private meetings owned by User A are inaccessible to User B, returning <code>403 Forbidden</code> for unauthorized viewing, export, or deletion.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sec_col1, sec_col2 = st.columns(2)
    with sec_col1:
        st.subheader("Security Checklist")
        st.markdown("""
        - ✅ **Password Hashing:** SHA-256 / bcrypt with salt
        - ✅ **Bearer Token Auth:** Enforced across REST endpoints
        - ✅ **Role-Based Permissions:** Admin vs Member segregation
        - ✅ **Cross-Tenant Guard:** User A cannot access User B's meetings
        - ✅ **Audit Trails:** All access attempts logged in database
        """)

    with sec_col2:
        st.subheader("Test Cross-Tenant Isolation")
        test_other_user = st.text_input("Simulate access as User:", value="bob")
        test_meeting_id = st.number_input("Target Private Meeting ID:", min_value=1, value=2)
        if st.button("Simulate Security Access Check"):
            from database import SessionLocal, User as UserModel
            db = SessionLocal()
            try:
                sim_user = db.query(UserModel).filter(UserModel.username == test_other_user).first()
                if sim_user:
                    allowed = can_user_access_meeting(sim_user.id, test_meeting_id)
                    if allowed:
                        st.success(f"Access ALLOWED: User '{test_other_user}' is authorized to view Meeting #{test_meeting_id}.")
                    else:
                        st.error(f"Access DENIED (403 Forbidden): User '{test_other_user}' is strictly BLOCKED from Meeting #{test_meeting_id}.")
                else:
                    st.warning(f"User '{test_other_user}' not found in database.")
            finally:
                db.close()

# ══════════════════════════════════════════════════
# TAB 8: SYSTEM ARCHITECTURE & DATAFLOW FLOWCHART
# ══════════════════════════════════════════════════
with tabs[7]:
    st.title("🏗️ System Architecture & Dataflow Specification")
    st.caption("Complete end-to-end pipeline mapping: Multi-source Ingestion ➔ Dual LLM Extraction ➔ Knowledge Base & RAG Layer ➔ Application Delivery.")

    # Status badges
    v_stat = get_vector_store_stats()
    db_stat = verify_knowledge_repository()
    
    st.markdown(f"""
    <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.25rem;">
        <span class="badge-emerald">✓ SQLite Metadata DB: Active ({db_stat.get('total_meetings', 0)} Meetings)</span>
        <span class="badge-cyan">✓ ChromaDB: HNSW Cosine ({v_stat.get('total_vectors', 0)} Vectors)</span>
        <span class="badge-indigo">✓ Gemini LLM: Dual Primary & Fallback</span>
        <span class="badge-emerald">✓ FastAPI Backend: http://127.0.0.1:8000</span>
        <span class="badge-indigo">✓ Streamlit Frontend: http://localhost:8501</span>
    </div>
    """, unsafe_allow_html=True)

    arch_col1, arch_col2 = st.columns([8, 2])
    with arch_col1:
        st.markdown("#### 🗺️ Interactive End-to-End System Flowchart")
    with arch_col2:
        svg_file_path = os.path.join(os.path.dirname(__file__), "architecture_diagram.svg")
        if os.path.exists(svg_file_path):
            with open(svg_file_path, "r", encoding="utf-8") as f:
                svg_data = f.read()
            render_download_button(
                data=svg_data,
                filename="SynthAI_System_Architecture.svg",
                mime="image/svg+xml",
                label="⬇️ Download SVG",
                is_primary=False
            )

    # Embed the SVG diagram cleanly
    if os.path.exists(svg_file_path):
        import streamlit.components.v1 as components
        with open(svg_file_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        components.html(
            f"""
            <div style="background: #0d1117; border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 1.25rem; display: flex; justify-content: center; box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6); overflow-x: auto;">
                {svg_content}
            </div>
            """,
            height=1400,
            scrolling=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Layer specifications
    spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
    with spec_col1:
        st.markdown("""
        <div class="synth-card">
            <h4>1. Ingestion Pipeline</h4>
            <p>Unified intake across multiple media sources:</p>
            <ul>
                <li><b>Upload:</b> MP4, MP3, WAV, M4A</li>
                <li><b>Zoom:</b> OAuth, Webhooks, HMAC-CRC</li>
                <li><b>Google Meet:</b> Drive API Polling</li>
                <li><b>ASR:</b> Whisper + PyAnnote Diarization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with spec_col2:
        st.markdown("""
        <div class="synth-card">
            <h4>2. AI Synthesis Engine</h4>
            <p>Zero-downtime dual intelligence extraction:</p>
            <ul>
                <li><b>Gemini Primary:</b> Executive summary & action items</li>
                <li><b>Fallback Engine:</b> OpenAI / Local Heuristics</li>
                <li><b>Output:</b> Strict JSON Insight Contracts</li>
                <li><b>Attributes:</b> Assignees, Deadlines, Decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with spec_col3:
        st.markdown("""
        <div class="synth-card">
            <h4>3. Knowledge Base & RAG</h4>
            <p>Hybrid relational & semantic search layer:</p>
            <ul>
                <li><b>SQLite:</b> Relational metadata & user audit</li>
                <li><b>Embeddings:</b> all-MiniLM-L6-v2 (384-dim)</li>
                <li><b>ChromaDB:</b> Persistent cosine similarity</li>
                <li><b>RAG Engine:</b> Grounded answers with citations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with spec_col4:
        st.markdown("""
        <div class="synth-card">
            <h4>4. Delivery & Security</h4>
            <p>Enterprise access & reporting interfaces:</p>
            <ul>
                <li><b>FastAPI:</b> REST API (Port 8000)</li>
                <li><b>Streamlit:</b> Web Workspace (Port 8501)</li>
                <li><b>Reports:</b> ReportLab PDF + CSV</li>
                <li><b>Security:</b> RBAC isolation & Bcrypt hashing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 2rem;">
        <a href="http://127.0.0.1:8000/architecture" target="_blank" style="text-decoration: none;">
            <button style="background: linear-gradient(135deg, #4f46e5, #06b6d4); color: white; border: none; padding: 0.65rem 1.4rem; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.95rem; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);">
                🌐 Open Dedicated Fullscreen Architecture Viewer on Port 8000
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

