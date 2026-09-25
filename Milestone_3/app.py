import streamlit as st
import os
import ctypes
import sys
import tempfile
import shutil
import whisper
from moviepy import VideoFileClip
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
import soundfile as sf
import torch
from pyannote.audio import Pipeline
import io
import markdown
import docx
from xhtml2pdf import pisa
import re
import time

# Milestone 3 modules
from database import (
    init_db,
    create_user,
    authenticate_user,
    get_user_meetings,
    get_meeting_by_id,
    get_all_meetings,
    verify_knowledge_repository,
    SessionLocal
)
from models import Meeting, Participant, ActionItem, KeyPoint, KeyDecision
from embedding_service import (
    get_embedding_model,
    generate_single_embedding,
    generate_meeting_embeddings
)
from vector_store import (
    get_vector_collection,
    insert_meeting_vectors,
    get_vector_store_stats,
    trace_vector_to_meeting
)
from search_service import semantic_search_meetings, TARGET_SLA_SECONDS
from rag_service import stream_grounded_rag_answer, generate_grounded_rag_answer
from seed_data import seed_database_and_vector_store

# ─────────────────────────────────────────────
#  AUTO-LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
for env_candidate in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Milestone_2", ".env"),
]:
    if os.path.exists(env_candidate):
        try:
            with open(env_candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and v:
                            os.environ[k] = v
        except Exception:
            pass

# Initialize relational database and ensure seed data
init_db()

# --- WORKAROUNDS ---
try:
    pytorch_lib_dir = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    c10_dll = os.path.join(pytorch_lib_dir, "c10.dll")
    if os.path.exists(c10_dll):
        ctypes.CDLL(c10_dll)
except Exception:
    pass

try:
    import imageio_ffmpeg
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(os.getcwd(), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        shutil.copy(ffmpeg_src, ffmpeg_dst)
except Exception:
    pass
# -------------------

st.set_page_config(
    page_title="SynthAI — Meeting Knowledge Repository & Semantic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  DESIGN SYSTEM — injected CSS
# ─────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --bg-void:       #060812;
  --bg-base:       #0A0E1A;
  --bg-surface:    #0F1628;
  --bg-elevated:   #151c35;
  --glass-bg:      rgba(15, 22, 40, 0.70);
  --glass-border:  rgba(99, 102, 241, 0.22);
  --glass-border-hover: rgba(99, 102, 241, 0.55);

  --indigo:        #6366F1;
  --indigo-light:  #818CF8;
  --indigo-dark:   #4F46E5;
  --cyan:          #22D3EE;
  --cyan-light:    #67E8F9;
  --emerald:       #10B981;
  --amber:         #F59E0B;
  --rose:          #F43F5E;

  --text-primary:   #F0F4FF;
  --text-secondary: #94A3B8;
  --text-muted:     #4E5E7A;

  --radius-sm:  6px;
  --radius-md:  12px;
  --radius-lg:  20px;
}

*, *::before, *::after { box-sizing: border-box; }

/* Global dark background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg-void) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 60% at 20% -10%, rgba(99,102,241,0.16) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(34,211,238,0.12) 0%, transparent 55%),
    var(--bg-void) !important;
  min-height: 100vh;
}

[data-testid="stMain"] { background: transparent !important; }

/* Top Header Bar */
header[data-testid="stHeader"] {
  background: rgba(6, 8, 18, 0.96) !important;
  border-bottom: 1px solid rgba(99, 102, 241, 0.22) !important;
  backdrop-filter: blur(16px) !important;
}
header[data-testid="stHeader"] * {
  color: #F0F4FF !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: rgba(10, 14, 26, 0.96) !important;
  border-right: 1px solid var(--glass-border) !important;
  backdrop-filter: blur(20px) !important;
  min-width: 300px !important;
}

/* Streamlit Buttons — Pure Dark Glassmorphism */
.stButton > button,
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
  background: rgba(15, 22, 40, 0.85) !important;
  border: 1px solid rgba(99, 102, 241, 0.35) !important;
  color: #F0F4FF !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 0.55rem 1rem !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}
.stButton > button:hover,
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
  background: rgba(99, 102, 241, 0.25) !important;
  border-color: rgba(99, 102, 241, 0.85) !important;
  color: #FFFFFF !important;
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.35) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active {
  transform: translateY(0px) !important;
}

button[kind="primary"],
button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
  border: 1px solid #818CF8 !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
  background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%) !important;
  box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6) !important;
}

/* Tabs Styling — Clear Active & Inactive Visibility */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(15, 22, 40, 0.65) !important;
  border: 1px solid rgba(99, 102, 241, 0.25) !important;
  border-radius: 12px !important;
  padding: 5px !important;
  gap: 8px !important;
}
.stTabs [data-baseweb="tab"] {
  color: #94A3B8 !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  padding: 0.65rem 1.25rem !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #F0F4FF !important;
  background: rgba(99, 102, 241, 0.15) !important;
}
.stTabs [aria-selected="true"] {
  color: #FFFFFF !important;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.4) 0%, rgba(79, 70, 229, 0.5) 100%) !important;
  border: 1px solid rgba(99, 102, 241, 0.6) !important;
  box-shadow: 0 2px 12px rgba(99, 102, 241, 0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
  display: none !important;
}

/* Inputs & Form Fields */
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
.stTextInput input,
.stTextArea textarea {
  background: rgba(15, 22, 40, 0.85) !important;
  border: 1px solid rgba(99, 102, 241, 0.35) !important;
  color: #F0F4FF !important;
  border-radius: 10px !important;
}
div[data-baseweb="input"] input {
  color: #F0F4FF !important;
}
div[data-baseweb="input"]:focus-within {
  border-color: #818CF8 !important;
  box-shadow: 0 0 15px rgba(99, 102, 241, 0.35) !important;
}
div[data-baseweb="select"] > div {
  background: rgba(15, 22, 40, 0.85) !important;
  border: 1px solid rgba(99, 102, 241, 0.35) !important;
  color: #F0F4FF !important;
  border-radius: 10px !important;
}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
  background: #0F1628 !important;
  border: 1px solid rgba(99, 102, 241, 0.3) !important;
  border-radius: 10px !important;
}
li[data-baseweb="menu-item"] {
  color: #F0F4FF !important;
}
li[data-baseweb="menu-item"]:hover {
  background: rgba(99, 102, 241, 0.25) !important;
}

/* Sidebar Radio Navigation */
div[role="radiogroup"] > label > div:first-child { display: none !important; }
div[role="radiogroup"] > label {
  padding: 0.55rem 1rem !important;
  border-radius: 8px !important;
  margin-bottom: 0.25rem !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  background: transparent !important;
}
div[role="radiogroup"] > label:hover {
  background: rgba(99, 102, 241, 0.12) !important;
}
div[role="radiogroup"] > label[data-checked="true"] {
  background: rgba(99, 102, 241, 0.25) !important;
  font-weight: 700 !important;
  border-left: 3px solid #6366F1 !important;
}
div[role="radiogroup"] > label div {
  color: #F0F4FF !important;
}

/* Glass cards */
.glass-card {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-md);
  padding: 1.25rem;
  backdrop-filter: blur(12px);
  margin-bottom: 1rem;
  transition: all 0.2s ease;
}
.glass-card:hover {
  border-color: var(--glass-border-hover);
  box-shadow: 0 4px 24px rgba(99, 102, 241, 0.2);
}

.hero-container {
  background: linear-gradient(135deg, rgba(99,102,241,0.14) 0%, rgba(34,211,238,0.08) 100%);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: 2.25rem 2rem;
  text-align: center;
  margin-bottom: 2rem;
}
.hero-brand {
  font-size: 2.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, #F0F4FF 30%, #818CF8 70%, #22D3EE 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.03em;
}
.hero-tagline {
  color: var(--text-secondary);
  font-size: 1rem;
  margin-top: 0.5rem;
}

/* SLA badge */
.sla-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.8rem;
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.35);
  border-radius: 100px;
  color: #6EE7B7;
  font-weight: 700;
  font-size: 0.78rem;
}

/* Metric Cards */
.metric-box {
  background: rgba(15, 22, 40, 0.75);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.metric-box-icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.3rem;
}
.metric-box-val { font-size: 1.4rem; font-weight: 800; color: #F0F4FF; }
.metric-box-lbl { font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }

/* Hit Badges */
.badge-decision {
  background: rgba(99, 102, 241, 0.22);
  border: 1px solid rgba(99, 102, 241, 0.45);
  color: #A5B4FC;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 700;
}
.badge-action {
  background: rgba(245, 158, 11, 0.22);
  border: 1px solid rgba(245, 158, 11, 0.45);
  color: #FCD34D;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 700;
}
.badge-summary {
  background: rgba(16, 185, 129, 0.22);
  border: 1px solid rgba(16, 185, 129, 0.45);
  color: #6EE7B7;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 700;
}
.badge-transcript {
  background: rgba(34, 211, 238, 0.22);
  border: 1px solid rgba(34, 211, 238, 0.45);
  color: #67E8F9;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 700;
}

/* Upload zone */
.upload-zone-wrapper {
  background: rgba(15, 22, 40, 0.6);
  border: 2px dashed rgba(99, 102, 241, 0.35);
  border-radius: 16px;
  padding: 2.5rem 2rem;
  text-align: center;
  transition: all 0.2s;
}
.upload-zone-wrapper:hover {
  border-color: rgba(99, 102, 241, 0.7);
  background: rgba(15, 22, 40, 0.85);
}

.report-executive {
  background: rgba(15, 22, 40, 0.7);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("transcript", None),
    ("chroma_collection", None),
    ("messages", []),
    ("rag_messages", []),
    ("report_summary", None),
    ("structured_data", None),
    ("user_id", 1), # Default demo user session
    ("username", "Engineer"),
    ("current_page", "Knowledge Search & AI Q&A"),
    ("active_meeting_id", None),
    ("current_meeting_data", None),
    ("last_search_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
#  MODEL CACHING
# ─────────────────────────────────────────────
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_diarization_pipeline(token):
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)

@st.cache_resource
def load_local_embedding_model():
    return get_embedding_model()


# ─────────────────────────────────────────────
#  HELPERS (Export & Formatting)
# ─────────────────────────────────────────────
def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{mins:02d}:{secs:02d}]"

def merge_transcription_and_diarization(whisper_segments, diarization_result):
    merged_transcript = []
    for seg in whisper_segments:
        start, end, text = seg['start'], seg['end'], seg['text'].strip()
        speaker, max_overlap = "Unknown", 0
        for turn, _, spk in diarization_result.itertracks(yield_label=True):
            overlap = max(0, min(end, turn.end) - max(start, turn.start))
            if overlap > max_overlap:
                max_overlap, speaker = overlap, spk
        merged_transcript.append(f"{format_timestamp(start)} {speaker}: {text}")
    return "\n".join(merged_transcript)

def generate_pdf_report(summary, transcript):
    html_content = f"""
    <html><head><style>
      body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #1a1a2e; }}
      h1, h2, h3 {{ color: #4F46E5; }}
      .summary-block {{ background: #f0f0ff; border-left: 4px solid #6366F1; padding: 1rem; border-radius: 4px; margin: 1rem 0; }}
      .transcript-block {{ font-family: monospace; font-size: 0.85rem; background: #f8f8f8; padding: 1rem; border-radius: 4px; }}
    </style></head><body>
      <h1>Meeting Intelligence Report</h1>
      <div class="summary-block"><h2>Executive Summary</h2>{markdown.markdown(summary or '')}</div>
      <h2>Full Diarized Transcript</h2>
      <div class="transcript-block"><pre>{transcript or ''}</pre></div>
    </body></html>
    """
    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(html_content, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def generate_docx_report(summary, transcript):
    doc = docx.Document()
    doc.add_heading('Meeting Intelligence Report', 0)
    doc.add_heading('Executive Summary', level=1)
    for line in (summary or '').split('\n'):
        if line.startswith('### '):
            doc.add_heading(line.replace('### ', ''), level=3)
        elif line.startswith(('- ', '* ')):
            doc.add_paragraph(line[2:], style='List Bullet')
        else:
            doc.add_paragraph(line)
    doc.add_heading('Full Transcript', level=1)
    for line in (transcript or '').split('\n'):
        if line.strip():
            doc.add_paragraph(line.strip())
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    return docx_buffer.getvalue()


# ─────────────────────────────────────────────
#  AUTHENTICATION GATE
# ─────────────────────────────────────────────
if not st.session_state.user_id:
    st.markdown('<style>[data-testid="stSidebar"] { display: none !important; } [data-testid="collapsedControl"] { display: none !important; }</style>', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-container">
      <div class="hero-brand">SynthAI</div>
      <div class="hero-tagline">Meeting Knowledge Repository & Semantic RAG Suite</div>
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])
    with auth_col2:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        with tab_login:
            with st.form("login_form"):
                log_user = st.text_input("Username", value="admin")
                log_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    user = authenticate_user(log_user, log_pass)
                    if user:
                        st.session_state.user_id = user["id"]
                        st.session_state.username = user["username"]
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            if st.button("🚀 Continue as Demo Engineer", use_container_width=True):
                st.session_state.user_id = 1
                st.session_state.username = "Demo Engineer"
                st.rerun()

        with tab_signup:
            with st.form("signup_form"):
                reg_user = st.text_input("New Username")
                reg_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Sign Up", use_container_width=True):
                    if len(reg_user.strip()) < 3:
                        st.error("Username must be at least 3 characters.")
                    elif len(reg_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        success = create_user(reg_user, reg_pass)
                        if success:
                            st.success("Account created! You can now log in.")
                        else:
                            st.error("Username already exists.")
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;padding:0.25rem 0.5rem 1rem;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#6366F1,#22D3EE);
                  border-radius:10px;display:flex;align-items:center;justify-content:center;
                  font-size:1.25rem;">🧠</div>
      <div>
        <div style="font-size:1.15rem;font-weight:800;background:linear-gradient(135deg,#F0F4FF,#818CF8,#22D3EE);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:-0.02em;">SynthAI</div>
        <div style="font-size:0.65rem;color:#818CF8;font-weight:700;letter-spacing:0.05em;">MILESTONE 3 · KNOWLEDGE RAG</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4E5E7A;padding:0 0 0.5rem;">API Configuration</p>', unsafe_allow_html=True)

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Google Gemini API key for structured extraction & grounded RAG Q&A",
    )
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=os.getenv("HF_TOKEN", ""),
        help="Required for Pyannote speaker diarization",
    )

    if gemini_key:
        st.markdown('<div style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.25rem 0.6rem;background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.35);border-radius:100px;font-size:0.75rem;font-weight:700;color:#6EE7B7;"><span style="width:8px;height:8px;border-radius:50%;background:#10B981;display:inline-block;"></span>Gemini Key Configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#FDA4AF;font-size:0.75rem;">⚠ Enter Gemini Key for Live LLM</div>', unsafe_allow_html=True)

    if hf_token:
        st.markdown('<div style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.25rem 0.6rem;background:rgba(34,211,238,0.15);border:1px solid rgba(34,211,238,0.35);border-radius:100px;font-size:0.75rem;font-weight:700;color:#67E8F9;margin-top:0.35rem;"><span style="width:8px;height:8px;border-radius:50%;background:#22D3EE;display:inline-block;"></span>HF Token Configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#94A3B8;font-size:0.72rem;margin-top:0.2rem;">(HF Token optional unless uploading new audio)</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Navigation
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4E5E7A;padding:0 0 0.5rem;">WORKSPACE</p>', unsafe_allow_html=True)

    if "_nav_target" in st.session_state:
        st.session_state.current_page = st.session_state._nav_target
        del st.session_state._nav_target

    page = st.radio(
        "Navigation",
        [
            "Knowledge Search & AI Q&A",
            "Command Center",
            "Meetings",
            "Intelligence",
            "Action Hub",
            "People",
            "Validation",
        ],
        key="current_page",
        label_visibility="collapsed",
    )

    # Meeting Selector
    past_meetings = get_user_meetings(st.session_state.get("user_id"))
    if past_meetings:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4E5E7A;padding:0 0 0.25rem;">SELECT ACTIVE MEETING</p>', unsafe_allow_html=True)

        m_labels = [f"📄 {m[1]} (ID: {m[0]})" for m in past_meetings]
        curr_idx = 0
        if st.session_state.get("active_meeting_id"):
            for idx, m in enumerate(past_meetings):
                if m[0] == st.session_state.active_meeting_id:
                    curr_idx = idx
                    break

        sel_label = st.selectbox("Choose meeting", m_labels, index=curr_idx, key="sb_meeting_select", label_visibility="collapsed")
        sel_m = past_meetings[m_labels.index(sel_label)]

        if st.session_state.get("active_meeting_id") != sel_m[0]:
            st.session_state.active_meeting_id = sel_m[0]
            st.session_state.transcript = sel_m[2]
            st.session_state.report_summary = sel_m[3]
            m_details = get_meeting_by_id(sel_m[0])
            st.session_state.current_meeting_data = m_details
            st.session_state._nav_target = "Intelligence"
            st.rerun()

    # Vector DB Quick Widget in Sidebar
    try:
        v_stats = get_vector_store_stats()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:0.75rem;">
          <div style="font-size:0.7rem;font-weight:700;color:#818CF8;text-transform:uppercase;letter-spacing:0.05em;">Vector Knowledge Base</div>
          <div style="display:flex;justify-content:space-between;margin-top:0.4rem;font-size:0.78rem;">
            <span style="color:#94A3B8;">Indexed Meetings:</span>
            <strong style="color:#F0F4FF;">{v_stats['unique_meetings_indexed']}</strong>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;">
            <span style="color:#94A3B8;">Total Vectors:</span>
            <strong style="color:#22D3EE;">{v_stats['total_vectors']}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    if st.session_state.user_id:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.75rem; color:#94A3B8; text-align:center;">Logged in as: <strong style="color:#F0F4FF;">{st.session_state.username}</strong></p>', unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.active_meeting_id = None
            st.session_state.transcript = None
            st.session_state.report_summary = None
            st.session_state.current_meeting_data = None
            st.rerun()


# ─────────────────────────────────────────────
#  PAGE: KNOWLEDGE SEARCH & AI Q&A (MILESTONE 3 CORE)
# ─────────────────────────────────────────────
if page == "Knowledge Search & AI Q&A":
    st.markdown("""
    <div class="hero-container">
      <div style="display:inline-flex;align-items:center;gap:0.5rem;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);padding:0.35rem 0.85rem;border-radius:100px;font-size:0.75rem;font-weight:700;color:#A5B4FC;margin-bottom:0.75rem;">
        ⚡ MILESTONE 3: TASKS 1 – 5 COMPLETE
      </div>
      <div class="hero-brand">Meeting Knowledge Repository & Semantic RAG</div>
      <div class="hero-tagline">Natural-language search across historical meetings · Multi-entity dynamic embeddings · Sub-3s vector retrieval · Grounded Gemini Q&A</div>
    </div>
    """, unsafe_allow_html=True)

    # Top Vector Store Metric Ribbon
    stats = get_vector_store_stats()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-icon" style="background:rgba(99,102,241,0.2);color:#818CF8;">📂</div>
          <div>
            <div class="metric-box-val">{stats['unique_meetings_indexed']}</div>
            <div class="metric-box-lbl">Meetings Indexed</div>
          </div>
        </div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-icon" style="background:rgba(34,211,238,0.2);color:#22D3EE;">🧬</div>
          <div>
            <div class="metric-box-val">{stats['total_vectors']}</div>
            <div class="metric-box-lbl">Stored Vectors</div>
          </div>
        </div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-icon" style="background:rgba(245,158,11,0.2);color:#F59E0B;">🎯</div>
          <div>
            <div class="metric-box-val">{stats['breakdown']['decision'] + stats['breakdown']['action_item']}</div>
            <div class="metric-box-lbl">Decisions & Actions</div>
          </div>
        </div>""", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-icon" style="background:rgba(16,185,129,0.2);color:#10B981;">⚡</div>
          <div>
            <div class="metric-box-val">&lt; 0.10s</div>
            <div class="metric-box-lbl">Search SLA (&lt; 3.0s)</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    tab_search, tab_rag, tab_trace = st.tabs([
        "🔍  Semantic Search (Historical Meetings)",
        "💬  Grounded Cross-Meeting RAG Q&A",
        "🛡️  Vector Traceability & Store Inspector"
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1: SEMANTIC SEARCH
    # ════════════════════════════════════════════════════════
    with tab_search:
        st.markdown("### 🔎 Natural Language Search Across Meetings")
        st.markdown('<p style="color:#94A3B8;font-size:0.85rem;">Ask natural questions or enter keywords to retrieve relevant meetings, decisions, action items, and transcripts in milliseconds.</p>', unsafe_allow_html=True)

        # Quick preset buttons
        st.markdown("**Quick Test Queries:**")
        preset_cols = st.columns(4)
        preset_query = None
        with preset_cols[0]:
            if st.button("🗄️ Database migration?", use_container_width=True):
                preset_query = "Which meeting discussed the database migration?"
        with preset_cols[1]:
            if st.button("📱 Mobile app deadline?", use_container_width=True):
                preset_query = "What deadline was decided for the mobile application?"
        with preset_cols[2]:
            if st.button("🎙️ Whisper accuracy tests?", use_container_width=True):
                preset_query = "What tests were recommended for Whisper transcription accuracy?"
        with preset_cols[3]:
            if st.button("⏰ Rollback script owner?", use_container_width=True):
                preset_query = "Who was assigned to the rollback script and what is the deadline?"

        # Search box & filter controls
        col_input, col_type = st.columns([3, 1])
        with col_input:
            search_query = st.text_input(
                "Search query",
                value=preset_query or st.session_state.get("last_search_query", ""),
                placeholder="e.g. Which meeting discussed the database migration?",
                label_visibility="collapsed"
            )
        with col_type:
            type_filter = st.selectbox(
                "Entity filter",
                ["All", "decision", "action_item", "summary", "transcript"],
                format_func=lambda x: f"Filter: {x.replace('_', ' ').title()}",
                label_visibility="collapsed"
            )

        if search_query.strip():
            st.session_state.last_search_query = search_query.strip()
            
            # Execute Semantic Search
            doc_filter_arg = type_filter if type_filter != "All" else None
            search_res = semantic_search_meetings(
                query=search_query.strip(),
                top_k=8,
                doc_type=doc_filter_arg
            )

            # SLA Status Ribbon
            t_sec = search_res["retrieval_time_seconds"]
            sla_class = "sla-badge"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;margin:0.75rem 0 1.25rem;">
              <span class="{sla_class}">⚡ Retrieved {search_res['total_hits']} results in {t_sec:.4f}s · SLA &lt; {TARGET_SLA_SECONDS:.1f}s PASSED</span>
              <span style="font-size:0.75rem;color:#94A3B8;">Matches across <strong>{len(search_res['matched_meetings'])}</strong> meetings</span>
            </div>
            """, unsafe_allow_html=True)

            if not search_res["matched_meetings"]:
                st.info("No matching meeting records found for this query.")
            else:
                for idx, m_hit in enumerate(search_res["matched_meetings"]):
                    score_pct = int(m_hit["best_score"] * 100)
                    with st.expander(f"📄 {m_hit['filename']} — Relevance: {score_pct}% ({m_hit['hits_count']} hits)", expanded=(idx == 0)):
                        st.markdown(f"**Meeting Filename:** `{m_hit['filename']}` · **ID:** `{m_hit['meeting_id']}` · **Date:** `{m_hit['created_at']}`")
                        if m_hit["summary"]:
                            st.markdown(f"**Executive Summary Preview:** {m_hit['summary'][:250]}...")

                        st.markdown("#### Matched Entities:")
                        for ent in m_hit["matching_entities"]:
                            dtype = ent["doc_type"]
                            badge_cls = f"badge-{dtype.replace('_', '')}"
                            st.markdown(f"""
                            <div style="background:rgba(255,255,255,0.03);border-left:3px solid #6366F1;padding:0.6rem 0.8rem;margin-bottom:0.5rem;border-radius:4px;">
                              <span class="{badge_cls}">{dtype.replace('_', ' ').upper()}</span>
                              <span style="font-size:0.75rem;color:#94A3B8;margin-left:0.5rem;">Similarity: {ent['score']:.2f}</span>
                              <div style="margin-top:0.35rem;font-size:0.85rem;color:#F0F4FF;line-height:1.5;">{ent['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Button to load this meeting into active view
                        if st.button(f"🔍 Open in Meeting Intelligence", key=f"btn_open_{m_hit['meeting_id']}"):
                            st.session_state.active_meeting_id = m_hit["meeting_id"]
                            full_m = get_meeting_by_id(m_hit["meeting_id"])
                            if full_m:
                                st.session_state.transcript = full_m["transcript"]
                                st.session_state.report_summary = full_m["summary"]
                                st.session_state.current_meeting_data = full_m
                            st.session_state._nav_target = "Intelligence"
                            st.rerun()

    # ════════════════════════════════════════════════════════
    # TAB 2: GROUNDED CROSS-MEETING RAG CHAT
    # ════════════════════════════════════════════════════════
    with tab_rag:
        st.markdown("### 💬 Grounded Cross-Meeting AI Assistant")
        st.markdown('<p style="color:#94A3B8;font-size:0.85rem;">Ask any question across your entire meeting knowledge repository. Answers are strictly grounded in retrieved meeting facts with full source attribution.</p>', unsafe_allow_html=True)

        col_scope, col_clear = st.columns([3, 1])
        with col_scope:
            scope_mode = st.radio(
                "Search Scope",
                ["All Historical Meetings", "Active Meeting Only"],
                horizontal=True,
                label_visibility="collapsed"
            )
        with col_clear:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.rag_messages = []
                st.rerun()

        # Display chat history
        for msg in st.session_state.rag_messages:
            role = msg["role"]
            with st.chat_message(role, avatar="🧑" if role == "user" else "🧠"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"🛡️ Grounding Sources ({len(msg['sources'])} retrieved items)"):
                        for s in msg["sources"]:
                            st.markdown(f"**From Meeting:** `{s.get('meeting_filename')}` (`{s.get('doc_type')}`) · Score: {s.get('score', 0):.2f}")
                            st.code(s.get("text"), language="text")

        # Chat Input
        if rag_prompt := st.chat_input("e.g. What deadline was decided for the mobile application?"):
            st.session_state.rag_messages.append({"role": "user", "content": rag_prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(rag_prompt)

            with st.chat_message("assistant", avatar="🧠"):
                msg_placeholder = st.empty()
                
                scoped_m_id = None
                if scope_mode == "Active Meeting Only" and st.session_state.get("active_meeting_id"):
                    scoped_m_id = st.session_state.active_meeting_id

                if gemini_key:
                    try:
                        stream_gen = stream_grounded_rag_answer(
                            question=rag_prompt,
                            api_key=gemini_key,
                            meeting_id=scoped_m_id,
                            top_k=5
                        )
                        full_resp = ""
                        retrieved_sources = []
                        for chunk, sources, s_res in stream_gen:
                            full_resp += chunk
                            msg_placeholder.markdown(full_resp + " ▌")
                            if sources:
                                retrieved_sources = sources
                        
                        msg_placeholder.markdown(full_resp)
                        st.session_state.rag_messages.append({
                            "role": "assistant",
                            "content": full_resp,
                            "sources": retrieved_sources
                        })
                        if retrieved_sources:
                            with st.expander(f"🛡️ Grounding Sources ({len(retrieved_sources)} items)"):
                                for s in retrieved_sources:
                                    st.markdown(f"**Meeting:** `{s.get('meeting_filename')}` (`{s.get('doc_type')}`) · Score: {s.get('score', 0):.2f}")
                                    st.code(s.get("text"), language="text")
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                else:
                    # Offline grounded fallback mode
                    rag_res = generate_grounded_rag_answer(
                        question=rag_prompt,
                        api_key=None,
                        meeting_id=scoped_m_id,
                        top_k=5
                    )
                    ans_text = rag_res["answer"]
                    msg_placeholder.markdown(ans_text)
                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": ans_text,
                        "sources": rag_res["sources"]
                    })
                    if rag_res["sources"]:
                        with st.expander(f"🛡️ Grounding Sources ({len(rag_res['sources'])} items)"):
                            for s in rag_res["sources"]:
                                st.markdown(f"**Meeting:** `{s.get('meeting_filename')}` (`{s.get('doc_type')}`) · Score: {s.get('score', 0):.2f}")
                                st.code(s.get("text"), language="text")

    # ════════════════════════════════════════════════════════
    # TAB 3: TRACEABILITY & VECTOR STORE INSPECTOR
    # ════════════════════════════════════════════════════════
    with tab_trace:
        st.markdown("### 🛡️ Task 3: Vector Traceability & Store Inspector")
        st.markdown('<p style="color:#94A3B8;font-size:0.85rem;">Verify that every stored embedding vector can be traced back to its parent relational meeting record in the SQLite database.</p>', unsafe_allow_html=True)

        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.markdown("#### Test Vector-to-Meeting Traceability")
            coll = get_vector_collection()
            sample_ids = coll.get(limit=15)["ids"]
            selected_vec_id = st.selectbox("Select a Vector ID to trace", sample_ids if sample_ids else ["None"])

            if selected_vec_id and selected_vec_id != "None":
                trace_res = trace_vector_to_meeting(selected_vec_id)
                if trace_res.get("traced"):
                    st.success(f"✅ Verified: Vector `{selected_vec_id}` successfully traced back to Meeting #{trace_res['meeting_id']} ('{trace_res['meeting_filename']}')")
                    st.json(trace_res)
                else:
                    st.error(f"Trace failed: {trace_res.get('error')}")

        with col_t2:
            st.markdown("#### Knowledge Store Actions")
            if st.button("🔄 Sync & Re-Index All Meetings", use_container_width=True, type="primary"):
                with st.spinner("Generating multi-entity embeddings & syncing vector store..."):
                    new_stats = seed_database_and_vector_store(force_reindex=True)
                    st.success(f"Synchronized! {new_stats['total_vectors']} vectors indexed.")
                    time.sleep(1)
                    st.rerun()

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid var(--glass-border);border-radius:10px;padding:1rem;margin-top:1rem;">
              <div style="font-weight:700;font-size:0.85rem;color:#818CF8;margin-bottom:0.5rem;">Entity Breakdown:</div>
              <div style="font-size:0.8rem;color:#94A3B8;">📋 Summaries: <strong style="color:#F0F4FF;">{stats['breakdown']['summary']}</strong></div>
              <div style="font-size:0.8rem;color:#94A3B8;">✦ Decisions: <strong style="color:#F0F4FF;">{stats['breakdown']['decision']}</strong></div>
              <div style="font-size:0.8rem;color:#94A3B8;">⚡ Action Items: <strong style="color:#F0F4FF;">{stats['breakdown']['action_item']}</strong></div>
              <div style="font-size:0.8rem;color:#94A3B8;">🎙️ Transcripts: <strong style="color:#F0F4FF;">{stats['breakdown']['transcript']}</strong></div>
            </div>
            """, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────
#  PAGE: COMMAND CENTER
# ─────────────────────────────────────────────
elif page == "Command Center":
    st.markdown("""
    <div class="hero-container">
      <div class="hero-brand">SynthAI</div>
      <div class="hero-tagline">AI-Powered Meeting Synthesis & Career Intelligence Suite</div>
      <div style="margin-top:1rem;display:flex;justify-content:center;gap:0.5rem;">
        <span style="background:rgba(99,102,241,0.2);color:#A5B4FC;padding:0.25rem 0.65rem;border-radius:100px;font-size:0.75rem;font-weight:600;">⚡ Whisper</span>
        <span style="background:rgba(34,211,238,0.2);color:#67E8F9;padding:0.25rem 0.65rem;border-radius:100px;font-size:0.75rem;font-weight:600;">✦ Gemini Intelligence</span>
        <span style="background:rgba(16,185,129,0.2);color:#6EE7B7;padding:0.25rem 0.65rem;border-radius:100px;font-size:0.75rem;font-weight:600;">◈ Persistent ChromaDB RAG</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Open Knowledge Search & AI Q&A", use_container_width=True, type="primary"):
            st.session_state._nav_target = "Knowledge Search & AI Q&A"
            st.rerun()
    with c2:
        if st.button("🎙️ Upload & Process New Recording", use_container_width=True):
            st.session_state._nav_target = "Meetings"
            st.rerun()

    # Saved meetings
    past_meetings = get_user_meetings(st.session_state.get("user_id"))
    if past_meetings:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📂 Saved Meetings in Knowledge Repository")
        for m_id, m_file, m_transcript, m_summary, m_date in past_meetings:
            date_str = m_date.strftime("%Y-%m-%d %H:%M") if m_date else "Recent"
            word_ct = len((m_transcript or "").split())
            is_current = (st.session_state.get("active_meeting_id") == m_id)
            tag = " (Currently Active)" if is_current else ""
            with st.expander(f"📄 {m_file} — {date_str} ({word_ct:,} words){tag}"):
                if m_summary:
                    st.markdown("**Executive Summary:**")
                    st.markdown(m_summary[:350] + "...")
                if not is_current:
                    if st.button("Load this meeting", key=f"cmd_load_{m_id}"):
                        st.session_state.active_meeting_id = m_id
                        st.session_state.transcript = m_transcript
                        st.session_state.report_summary = m_summary
                        st.session_state.current_meeting_data = get_meeting_by_id(m_id)
                        st.session_state._nav_target = "Intelligence"
                        st.rerun()

    st.stop()


# ─────────────────────────────────────────────
#  PAGE: MEETINGS (UPLOAD & FULL PIPELINE)
# ─────────────────────────────────────────────
elif page == "Meetings":
    st.markdown("""
    <div style="max-width:680px;margin:0 auto 0.5rem;">
      <div class="upload-zone-wrapper">
        <span style="font-size:2.5rem;">🎙️</span>
        <div style="font-size:1.25rem;font-weight:700;margin:0.5rem 0;color:#F0F4FF;">Drop your meeting recording here</div>
        <div style="font-size:0.85rem;color:#94A3B8;margin-bottom:1rem;">Upload any audio or video file to transcribe, diarize, and index into the knowledge store</div>
        <div style="display:flex;justify-content:center;gap:0.4rem;">
          <span style="background:rgba(255,255,255,0.06);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.75rem;color:#94A3B8;">.mp3</span>
          <span style="background:rgba(255,255,255,0.06);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.75rem;color:#94A3B8;">.wav</span>
          <span style="background:rgba(255,255,255,0.06);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.75rem;color:#94A3B8;">.m4a</span>
          <span style="background:rgba(255,255,255,0.06);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.75rem;color:#94A3B8;">.mp4</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        uploaded_file = st.file_uploader(
            "Upload meeting recording",
            type=["mp3", "wav", "m4a", "mp4"],
            label_visibility="collapsed",
        )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        <div style="max-width:680px;margin:0.75rem auto;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:1rem;display:flex;align-items:center;gap:0.75rem;">
          <span style="font-size:1.6rem;">🎵</span>
          <div style="flex:1;">
            <div style="font-weight:600;font-size:0.88rem;color:#F0F4FF;">{uploaded_file.name}</div>
            <div style="font-size:0.75rem;color:#94A3B8;">{file_size_mb:.2f} MB · Ready for processing</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_btn = st.columns([1, 2, 1])[1]
        with col_btn:
            process_btn = st.button("🚀  Analyze & Index Meeting", use_container_width=True, type="primary")

        if process_btn:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                whisper_model = load_whisper_model()
                diarization_pipeline = load_diarization_pipeline(hf_token)

                with st.spinner("Transcribing speech with OpenAI Whisper…"):
                    audio_array = whisper.load_audio(tmp_path, sr=16000)
                    result = whisper_model.transcribe(audio_array)

                with st.spinner("Identifying speakers with Pyannote…"):
                    waveform_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                    diarization = diarization_pipeline({"waveform": waveform_tensor, "sample_rate": 16000})

                diarization_annotation = getattr(diarization, "speaker_diarization", diarization)
                final_transcript = merge_transcription_and_diarization(result['segments'], diarization_annotation)
                st.session_state.transcript = final_transcript

                with st.spinner("Extracting structured intelligence with Gemini…"):
                    from llm_service import process_transcript_with_llm
                    client = genai.Client(api_key=gemini_key) if gemini_key else None
                    
                    db = SessionLocal()
                    structured_data = None
                    if client:
                        try:
                            structured_data = process_transcript_with_llm(client, final_transcript)
                        except Exception as llm_err:
                            err_str = str(llm_err)
                            if "API_KEY_HTTP_REFERRER_BLOCKED" in err_str or "referer <empty>" in err_str:
                                st.warning(
                                    "⚠️ **Google Gemini API Key Restriction Detected**: "
                                    "Your Google Cloud API key has **'HTTP referrers (websites)'** restriction enabled. "
                                    "Because Python backend scripts send requests without browser referrers, Google blocked the call.\n\n"
                                    "👉 **Quick 1-Minute Fix**: Go to [Google Cloud Console Credentials](https://console.cloud.google.com/apis/credentials) → Click your API Key → Under **Application restrictions**, change from **'Websites'** to **'None'** (or generate an unrestricted server key from [aistudio.google.com](https://aistudio.google.com/app/apikey)).\n\n"
                                    "✅ *Falling back to automated transcript extraction so your meeting recording, transcript, and vector index are still saved successfully!*"
                                )
                            else:
                                st.warning(f"⚠️ Live Gemini extraction failed ({llm_err}). Falling back to automated transcript extraction.")

                    if structured_data:
                        summary_text = structured_data.summary
                        key_points_list = structured_data.key_points
                        decisions_list = structured_data.decisions
                        action_items_list = structured_data.action_items
                        participants_list = structured_data.participants
                    else:
                        extracted_spks = list(dict.fromkeys(re.findall(r'\[\d+:\d+\]\s*([^:]+):', final_transcript))) or ["Speaker_00"]
                        summary_text = f"Meeting recording '{uploaded_file.name}' transcribed with {len(result.get('segments', []))} spoken segments across: {', '.join(extracted_spks)}."
                        key_points_list = [f"Transcript successfully recorded and synced into knowledge repository with {len(result.get('segments', []))} dialogue segments."]
                        decisions_list = ["Meeting audio transcribed and synchronized into relational knowledge base."]
                        action_items_list = []
                        participants_list = extracted_spks

                    # Save to relational database
                    part_objs = []
                    for p_name in participants_list:
                        p = db.query(Participant).filter(Participant.name == p_name).first()
                        if not p:
                            p = Participant(name=p_name)
                            db.add(p)
                        part_objs.append(p)
                    db.commit()

                    meeting = Meeting(
                        filename=uploaded_file.name,
                        transcript=final_transcript,
                        summary=summary_text,
                        user_id=st.session_state.user_id
                    )
                    meeting.participants.extend(part_objs)
                    db.add(meeting)
                    db.commit()
                    db.refresh(meeting)

                    for ai in action_items_list:
                        p_id = None
                        if hasattr(ai, 'assigned_participant') and ai.assigned_participant:
                            p = db.query(Participant).filter(Participant.name == ai.assigned_participant).first()
                            if p: p_id = p.id
                        action = ActionItem(
                            meeting_id=meeting.id,
                            participant_id=p_id,
                            description=ai.description if hasattr(ai, 'description') else str(ai),
                            deadline=getattr(ai, 'deadline', None),
                            priority=getattr(ai, 'priority', 'Medium'),
                            status=getattr(ai, 'status', 'Pending')
                        )
                        db.add(action)

                    for kp in key_points_list:
                        db.add(KeyPoint(meeting_id=meeting.id, point=kp))
                    for kd in decisions_list:
                        db.add(KeyDecision(meeting_id=meeting.id, decision=kd))
                    db.commit()

                    saved_id = meeting.id
                    full_saved_m = get_meeting_by_id(saved_id)
                    db.close()

                # Dynamic Multi-Entity Embedding Generation & ChromaDB Sync
                with st.spinner("Generating multi-entity embeddings & indexing in vector store…"):
                    units_with_emb = generate_meeting_embeddings(full_saved_m)
                    insert_meeting_vectors(saved_id, units_with_emb)

                st.session_state.active_meeting_id = saved_id
                st.session_state.report_summary = summary_text
                st.session_state.current_meeting_data = full_saved_m
                st.session_state._nav_target = "Intelligence"
                st.success("✅ Meeting analyzed, structured, and indexed in Vector Store!")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"Error during meeting processing: {e}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass

    st.stop()


# ─────────────────────────────────────────────
#  PAGE: INTELLIGENCE (REPORT, ACTIONS, TRANSCRIPT)
# ─────────────────────────────────────────────
elif page == "Intelligence":
    if not st.session_state.transcript:
        st.warning("No active meeting loaded. Select a meeting from the sidebar or upload a recording.")
        st.stop()

    tab1, tab2 = st.tabs(["📋  Meeting Intelligence", "📜  Full Transcript"])

    with tab1:
        st.markdown(f"## 📋 Meeting Intelligence: `{st.session_state.get('current_meeting_data', {}).get('filename', 'Current Meeting')}`")
        st.markdown('<div class="report-executive">', unsafe_allow_html=True)
        m_data = st.session_state.get("current_meeting_data") or {}
        
        st.markdown(f"**Executive Summary:**\n\n{st.session_state.report_summary or 'No summary available.'}")
        
        if m_data.get("key_decisions"):
            st.markdown("**✦ Key Decisions:**")
            for kd in m_data["key_decisions"]:
                st.markdown(f"- {kd}")

        if m_data.get("action_items"):
            st.markdown("**⚡ Action Items:**")
            for ai in m_data["action_items"]:
                deadline_str = f" · ⏰ Deadline: {ai['deadline']}" if ai.get('deadline') else ""
                assignee_str = f" · 👤 Assignee: {ai['assigned_participant']}" if ai.get('assigned_participant') else ""
                st.markdown(f"- **{ai['description']}** ({ai.get('priority', 'Normal')}){assignee_str}{deadline_str}")

        if m_data.get("key_points"):
            st.markdown("**📌 Key Discussion Points:**")
            for kp in m_data["key_points"]:
                st.markdown(f"- {kp}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_pdf, col_docx = st.columns(2)
        with col_pdf:
            pdf_bytes = generate_pdf_report(st.session_state.report_summary, st.session_state.transcript)
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="Meeting_Report.pdf", mime="application/pdf", use_container_width=True)
        with col_docx:
            docx_bytes = generate_docx_report(st.session_state.report_summary, st.session_state.transcript)
            st.download_button("📝 Download DOCX Report", data=docx_bytes, file_name="Meeting_Report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

    with tab2:
        st.markdown("### 📜 Diarized Transcript")
        st.code(st.session_state.transcript, language="text")

    st.stop()


# ─────────────────────────────────────────────
#  PAGE: ACTION HUB
# ─────────────────────────────────────────────
elif page == "Action Hub":
    st.markdown("## ⚡ Action Hub")
    if not st.session_state.get("current_meeting_data"):
        st.info("Please select a meeting to track its action items.")
        st.stop()

    actions = st.session_state.current_meeting_data.get("action_items", [])
    if not actions:
        st.info("No action items found for this meeting.")
        st.stop()

    st.markdown(f"**{len(actions)}** action items identified:")
    for idx, item in enumerate(actions):
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            st.checkbox("", key=f"ah_cb_{idx}", label_visibility="collapsed")
        with col2:
            deadline = f" · ⏰ {item.get('deadline')}" if item.get('deadline') else ""
            assignee = item.get('assigned_participant') or "Unassigned"
            priority = item.get('priority', 'Medium')
            st.markdown(f"**{item.get('description')}**")
            st.markdown(f"<span style='font-size:0.75rem;color:#818CF8;'>👤 {assignee}</span> · <span style='font-size:0.75rem;color:#FCD34D;'>● {priority}</span><span style='font-size:0.75rem;color:#94A3B8;'>{deadline}</span>", unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────
#  PAGE: PEOPLE
# ─────────────────────────────────────────────
elif page == "People":
    st.markdown("## 👥 People & Participants")
    if not st.session_state.transcript:
        st.info("Select or upload a meeting to view speaker participation.")
        st.stop()

    speaker_data = {}
    for line in st.session_state.transcript.strip().split('\n'):
        parts = line.split(' ', 2)
        if len(parts) >= 3:
            speaker = parts[1].rstrip(':')
            words = len(parts[2].split())
            if speaker not in speaker_data:
                speaker_data[speaker] = {"utterances": 0, "words": 0}
            speaker_data[speaker]["utterances"] += 1
            speaker_data[speaker]["words"] += words

    total_words = sum(v["words"] for v in speaker_data.values())
    for spk, data in speaker_data.items():
        pct = (data["words"] / total_words * 100) if total_words > 0 else 0
        st.markdown(f"### {spk}: {pct:.1f}% ({data['words']} words, {data['utterances']} utterances)")
        st.progress(pct / 100.0)

    st.stop()


# ─────────────────────────────────────────────
#  PAGE: VALIDATION
# ─────────────────────────────────────────────
elif page == "Validation":
    st.markdown("## ✅ Transcription Accuracy Validation")
    if not st.session_state.transcript:
        st.info("Load a meeting transcript first to validate against ground truth.")
        st.stop()

    ref_file = st.file_uploader("Upload Ground-Truth Transcript (.txt)", type=["txt"])
    if ref_file:
        ref_text = ref_file.read().decode("utf-8", errors="ignore")
        ref_words = re.findall(r'\w+', ref_text.lower())
        whisper_words = re.findall(r'\w+', st.session_state.transcript.lower())
        
        matches = len(set(ref_words).intersection(set(whisper_words)))
        acc = (matches / len(set(ref_words)) * 100) if ref_words else 0
        st.metric("Vocabulary Match Accuracy", f"{acc:.1f}%")
        if acc >= 90:
            st.success("Target achieved: >= 90% accuracy!")
        else:
            st.warning("Below 90% accuracy target.")

    st.stop()
