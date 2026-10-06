<div align="center">

# 🧠 SynthAI — Enterprise Career & Meeting Intelligence Suite

### *Autonomous Multi-Modal Meeting Transcription, Semantic Vector Knowledge Graph, Cloud Integrations & Grounded RAG Platform*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://careerintelligance-eedryyrvermedn5wsjyvdx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%20–%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-fc60a8?style=for-the-badge)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **Transform unstructured audio/video meetings into enterprise-grade, actionable intelligence.**
> From raw recordings to speaker diarization, persistent vector knowledge graphs, sub-3-second cross-meeting semantic search, cloud sync (Zoom & Google Meet), PDF/CSV reporting, and containerized deployment.

<br/>

</div>

---

## 📑 Table of Contents

- [✨ Milestone Roadmap](#-milestone-roadmap)
- [🏗️ Complete System Architecture](#️-complete-system-architecture)
- [🧠 Milestone 1: Meeting Intelligence Pipeline](#-milestone-1--synthai-meeting-intelligence-pipeline)
- [📊 Milestone 2: Relational Database & Structured Analytics](#-milestone-2--relational-database--structured-analytics)
- [🔍 Milestone 3: Knowledge Repository & Semantic RAG](#-milestone-3--knowledge-repository--semantic-rag)
- [🚀 Milestone 4: Enterprise Dashboard, Cloud Integrations & Production Deployment](#-milestone-4--enterprise-dashboard-cloud-integrations--production-deployment)
- [🔌 REST API Reference](#-rest-api-reference)
- [🖥️ Quick Start & Local Setup](#️-quick-start--local-setup)
- [🐳 Docker & Container Orchestration](#-docker--container-orchestration)
- [🧪 Test Suites & Quality Assurance](#-test-suites--quality-assurance)
- [🗂️ Complete Repository Structure](#️-complete-repository-structure)
- [🤝 Contributing & License](#-contributing--license)

---

## ✨ Milestone Roadmap

This repository is architected across four progressive milestones, culminating in a full-scale enterprise AI platform:

| Milestone | Application | Focus & Capabilities | Tech Stack | Status |
|:---:|:---|:---|:---|:---:|
| **[Milestone 1](#-milestone-1--synthai-meeting-intelligence-pipeline)** | **SynthAI Core** | Audio extraction, Whisper transcription, Pyannote diarization, single-meeting Gemini summary & RAG chat | Whisper, Pyannote 3.1, Gemini Flash, ChromaDB, Streamlit | ✅ [Live](https://careerintelligance-eedryyrvermedn5wsjyvdx.streamlit.app/) |
| **[Milestone 2](#-milestone-2--relational-database--structured-analytics)** | **Structured Knowledge DB** | Pydantic JSON extraction, SQLite relational persistence, action item assignment, priority & status tracking | SQLAlchemy, SQLite, Pydantic, Gemini Flash, Streamlit | ✅ Complete |
| **[Milestone 3](#-milestone-3--knowledge-repository--semantic-rag)** | **Semantic Knowledge Repository** | Cross-meeting multi-entity vector embeddings, persistent ChromaDB, sub-3s semantic search SLA, grounded RAG Q&A with source citations, FastAPI REST API | ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`), FastAPI, Gemini | ✅ Complete |
| **[Milestone 4](#-milestone-4--enterprise-dashboard-cloud-integrations--production-deployment)** | **Enterprise Suite & Deployment** | 7-tab modern UI, Zoom Cloud Sync, Google Meet / Drive pipeline, Plotly analytics, PDF/CSV report generation, multi-tenant RBAC security, Docker deployment | FastAPI, Streamlit, ReportLab, Plotly, Docker, Docker-Compose | ✅ Complete |

---

## 🏗️ Complete System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                  Input Ingestion Layer                 │
                                  │   Local Upload (.mp4/.wav/.mp3) • Zoom API • Google Meet│
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │            Audio Extraction & Transcription            │
                                  │      FFmpeg Audio Extraction  ──▶  OpenAI Whisper      │
                                  │      Pyannote Speaker Diarization (Who Spoke When)     │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │          Structured Intelligence Engine (LLM)          │
                                  │       Google Gemini Flash (JSON Structured Extraction) │
                                  │     • Executive Summary  • Decisions  • Action Items   │
                                  └─────────────────────┬──────────────────┬───────────────┘
                                                        │                  │
                         ┌──────────────────────────────┘                  └──────────────────────────────┐
                         ▼                                                                                ▼
┌─────────────────────────────────────────────────┐                             ┌─────────────────────────────────────────────────┐
│           Relational Knowledge Base             │                             │           Vector Knowledge Graph                │
│             (SQLite / SQLAlchemy)               │                             │             (Persistent ChromaDB)               │
│  • Meetings & Multi-Tenant User Isolation       │                             │  • 384-d Dense SentenceTransformers Embeddings  │
│  • Action Items (Status, Priority, Assignee)    │                             │  • Multi-Entity Indexing (Decisions, Summaries) │
│  • Decisions, Sync Logs & Security Audit Trails │                             │  • Cosine Similarity & Metadata Filtering       │
└────────────────────────┬────────────────────────┘                             └────────────────────────┬────────────────────────┘
                         │                                                                               │
                         └──────────────────────────────┬────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │                 Enterprise API & Engine                │
                                  │  FastAPI (Port 8000) • Sub-3s SLA Search • Grounded RAG│
                                  │  Zoom/Meet Ingestion Pipelines • ReportLab PDF & CSV   │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │               Modern Web Presentation UI               │
                                  │      Streamlit Dashboard (Port 8501) • 7 Active Tabs   │
                                  │     Plotly Visualizations • Source Evidence Explorer   │
                                  └────────────────────────────────────────────────────────┘
```

---

## 🧠 Milestone 1 — SynthAI: Meeting Intelligence Pipeline

> 📂 Directory: [`Milestone_1/`](Milestone_1/)  
> 🔗 **[Try the Live Web App on Streamlit Cloud →](https://careerintelligance-eedryyrvermedn5wsjyvdx.streamlit.app/)**

Milestone 1 introduces the core audio-to-intelligence pipeline, handling media extraction, speech-to-text, speaker attribution, structured summarization, and interactive RAG chat for single meetings.

### 🔄 Ingestion & Diarization Flow
1. **Audio Extraction:** Uses FFmpeg to extract high-fidelity 16kHz WAV mono audio from any uploaded media format (`.mp4`, `.mov`, `.mkv`, `.wav`, `.mp3`).
2. **Transcription:** OpenAI Whisper generates time-stamped text segments with punctuation and multi-lingual support.
3. **Speaker Diarization:** Pyannote Audio 3.1 creates a precise acoustic speaker timeline identifying who was speaking during every interval.
4. **Alignment Merger:** The transcript and diarization timelines are aligned to produce a clean, speaker-attributed dialogue script (*"Speaker 0: Hello team..."*).
5. **Generative Intelligence:** Google Gemini Flash crafts an executive summary, high-impact decisions, and designated action items.
6. **In-Memory RAG:** Sentence-Transformers and ChromaDB enable real-time Q&A against the meeting transcript.

### 🛠️ Key Technologies
- **UI:** Streamlit
- **ASR & Diarization:** OpenAI Whisper + Pyannote Audio 3.1
- **LLM:** Google Gemini Flash (`gemini-1.5-flash` / `gemini-2.0-flash`)
- **Document Export:** `xhtml2pdf` (PDF), `python-docx` (Word), `.txt`

---

## 📊 Milestone 2 — Relational Database & Structured Analytics

> 📂 Directory: [`Milestone_2/`](Milestone_2/)

Milestone 2 transitions SynthAI from ephemeral session-based outputs into a persistent, relational meeting repository with structured data modeling and tracking.

### 🌟 Key Capabilities
- **Strict Pydantic Schemas:** Forces Gemini LLM to respond in strictly validated JSON schemas, ensuring zero formatting failures for summaries, action items, and participants.
- **Relational Storage:** SQLite database backed by SQLAlchemy ORM with foreign keys connecting meetings, action items, and participants.
- **Action Item Tracking:** Tracks task owners, priorities (`High`, `Medium`, `Low`), deadlines, and lifecycle statuses (`Pending`, `In Progress`, `Completed`).
- **Interactive Management UI:** Edit action item status, filter by participant or priority, and view relational statistics directly in Streamlit.

---

## 🔍 Milestone 3 — Knowledge Repository & Semantic RAG

> 📂 Directory: [`Milestone_3/`](Milestone_3/)

Milestone 3 evolves the platform into an enterprise cross-meeting knowledge base. Historical meetings are structured across relational entities and synchronized into a high-performance vector store, enabling sub-3-second cross-meeting semantic search and strictly grounded RAG Q&A.

### 🚀 Key Task Breakdown
- **Task 1 — Meeting Knowledge Repository:** Organizes historical meetings into structured knowledge units (metadata, summaries, decisions, action items, dialogue turns).
- **Task 2 — Multi-Entity Dynamic Embeddings:** Generates dense 384-dimensional embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) across distinct entity types.
- **Task 3 — Persistent ChromaDB Integration:** Disk-persisted vector storage with CRUD operations, cosine similarity search, and entity metadata filtering (`doc_type`, `meeting_id`).
- **Task 4 — Sub-3s Semantic Search:** Natural-language search across entire meeting histories (*"Which meeting discussed database migration?"*) with dynamic ranking and strict SLA verification (~0.06s typical).
- **Task 5 — Grounded RAG Q&A with Citations:** Multi-turn AI question-answering with Google Gemini, anchored exclusively to retrieved meeting snippets with verifiable evidence source cards.
- **Task 6 — FastAPI REST API Layer:** Production REST endpoints (`/meetings`, `/meetings/{id}`, `/search`, `/ask`) with JWT Bearer authentication and OpenAPI documentation.
- **Tasks 7–10 — Stress & Edge-Case Testing:** Verified fault tolerance on 150+ dialogue turns, irrelevant query handling, and automated end-to-end test execution.

---

## 🚀 Milestone 4 — Enterprise Dashboard, Cloud Integrations & Production Deployment

> 📂 Directory: [`Milestone_4/`](Milestone_4/)  
> 📖 Architectural Guide: [`Milestone_4/ARCHITECTURE.md`](Milestone_4/ARCHITECTURE.md)

Milestone 4 unifies the entire SynthAI ecosystem into a production-grade, multi-tenant, cloud-integrated platform with deep analytics, reporting, and containerization.

### 🖥️ 7-Tab Streamlit Dashboard
1. **🏠 Executive Dashboard:** High-level platform KPIs (total meetings, logged hours, completed vs open action items, participant counts).
2. **📤 Upload & Ingestion:** Drag-and-drop audio/video processing with auto-transcription, diarization, database persistence, and vector indexing.
3. **📁 Meeting Explorer:** Deep-dive viewer featuring speaker dialogue transcript, Gemini summary, decisions list, action items table, and direct exports.
4. **🔄 Cloud Integrations:** Direct Zoom Cloud Recording and Google Meet (Google Drive) sync with duplicate suppression.
5. **🔍 Semantic Search & RAG:** Real-time cross-meeting natural language query search with latency SLA counters and grounded Gemini AI answers.
6. **📄 Report Center:** One-click download of styled ReportLab PDF briefs and RFC-4180 CSV exports.
7. **📊 Analytics & Insights:** Interactive Plotly charts analyzing action item priority distributions, completion ratios, and participant workload.

### 🔒 Enterprise Security & Multi-Tenancy
- **Cross-Tenant Isolation:** Users can only query, view, export, or modify meetings belonging to their tenant organization (`403 Forbidden` enforcement).
- **Password Security:** Salted bcrypt password hashing (`passlib[bcrypt]`).
- **Audit Trails:** Relational security audit logging recording authentication attempts, uploads, sync events, and denied access attempts.

### 📥 Native Cloud Pipelines
- **Zoom Cloud Integration:** Ingests recordings using Zoom Cloud APIs / Webhooks with HMAC-SHA256 signature verification and SHA-256 duplicate content prevention.
- **Google Meet Sync:** Polls and downloads recorded sessions from Google Drive, automatically processing and indexing them into the vector graph.

### 📄 High-Fidelity Reports & Exports
- **Executive PDF Briefs:** Custom ReportLab generator with branded header, metadata overview, key decisions, and formatted action item tables.
- **RFC-4180 CSV Reports:** Full tabular export containing relational meeting attributes.

---

## 🔌 REST API Reference

The FastAPI backend runs on `http://localhost:8000` with interactive Swagger docs at `/docs`.

| Endpoint | Method | Description | Auth Required |
|---|:---:|---|:---:|
| `/health` | `GET` | System health and vector store readiness | No |
| `/auth/register` | `POST` | Register a new user | No |
| `/auth/login` | `POST` | Authenticate user and receive JWT access token | No |
| `/meetings` | `GET` | List all accessible meetings with filters | Yes |
| `/meetings/{id}` | `GET` | Get full meeting intelligence details | Yes |
| `/search` | `POST` | Sub-3s cross-meeting semantic search with SLA score | Yes |
| `/ask` | `POST` | Grounded RAG question-answering with evidence sources | Yes |
| `/integrations/zoom/sync` | `POST` | Ingest Zoom recording with duplicate protection | Yes |
| `/integrations/meet/sync` | `POST` | Ingest Google Meet recording from Drive | Yes |
| `/reports/pdf/{id}` | `GET` | Download executive ReportLab PDF report | Yes |
| `/reports/csv/{id}` | `GET` | Download RFC-4180 structured CSV report | Yes |
| `/analytics/summary` | `GET` | Retrieve aggregate metrics and workload stats | Yes |

---

## 🖥️ Quick Start & Local Setup

### 1. Prerequisites
- **Python:** 3.9 – 3.11 installed
- **FFmpeg:** Installed and added to system `PATH` (bundled executable included for Windows)
- **API Keys:**
  - [Google Gemini API Key](https://aistudio.google.com/) (Required for summarization & RAG)
  - [HuggingFace Token](https://huggingface.co/settings/tokens) (Required for Pyannote diarization in Milestone 1)

### 2. Clone the Repository
```bash
git clone https://github.com/Nipunbhadane123/Career_intelligance.git
cd Career_intelligance
```

### 3. Launch by Milestone

#### 🚀 Running Milestone 4 (Full Enterprise Suite)
```bash
cd Milestone_4
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
cp .env.example .env          # Set your GEMINI_API_KEY
```

**One-Click Windows Launch:**
```cmd
run_all.bat
```
*(Or run separately: `run_backend.bat` on Port 8000 and `run_frontend.bat` on Port 8501)*

#### 🔍 Running Milestone 3 (Knowledge Repository & API)
```bash
cd Milestone_3
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
streamlit run app.py
```

#### 🧠 Running Milestone 1 (Core Synthesizer)
```bash
cd Milestone_1
pip install -r requirements.txt
streamlit run app.py
```

---

## 🐳 Docker & Container Orchestration

Run the complete Milestone 4 production stack with a single command:

```bash
cd Milestone_4
docker-compose up --build
```

- **Backend REST API:** `http://localhost:8000`
- **Streamlit Dashboard:** `http://localhost:8501`
- **Health Check:** `http://localhost:8000/health`
- **Persistent Storage:** SQLite database and ChromaDB vector index are automatically persisted across container restarts via Docker named volumes.

---

## 🧪 Test Suites & Quality Assurance

Each milestone includes rigorous unit, integration, security, and performance test suites:

```bash
# Run Milestone 4 Consolidated Verification Suite
cd Milestone_4
python test_milestone4.py

# Run Multi-Tenant Security & 403 Forbidden Isolation Tests
python test_task_7_security.py

# Run Cloud Integrations & Duplicate Protection Tests
python test_tasks_4_to_5_integrations.py

# Run PDF & CSV Report Fidelity Tests
python test_task_6_reports.py

# Run End-to-End Autonomous Pipeline Tests
python test_task_8_end_to_end.py

# Run Latency SLA & Reliability Tests
python test_task_9_performance_and_reliability.py
```

---

## 🗂️ Complete Repository Structure

```text
Career_intelligance/
│
├── Milestone_1/                        # Core Audio & Meeting Synthesizer
│   ├── app.py                          # Streamlit application with Whisper + Pyannote + Gemini
│   ├── requirements.txt                # Dependencies (Whisper, Pyannote, Gemini, ChromaDB)
│   ├── ffmpeg.exe                      # Windows media processing binary
│   └── README.md                       # Milestone 1 architecture & pipeline docs
│
├── Milestone_2/                        # Structured Database & Action Item Tracking
│   ├── app.py                          # Streamlit dashboard with SQLite views
│   ├── database.py                     # SQLite engine & CRUD operations
│   ├── models.py                       # SQLAlchemy ORM models (Meetings, ActionItems)
│   ├── schemas.py                      # Pydantic schemas for structured extraction
│   ├── llm_service.py                  # Structured Gemini extraction service
│   └── README.md                       # Milestone 2 documentation
│
├── Milestone_3/                        # Semantic Knowledge Repository & REST API
│   ├── api.py                          # FastAPI REST API (/meetings, /search, /ask)
│   ├── app.py                          # Streamlit knowledge explorer
│   ├── vector_store.py                 # Persistent ChromaDB integration
│   ├── embedding_service.py            # SentenceTransformers multi-entity embedder
│   ├── search_service.py               # Sub-3s SLA semantic search engine
│   ├── rag_service.py                  # Grounded Gemini RAG pipeline with citations
│   ├── database.py                     # Relational knowledge repository
│   ├── seed_data.py                    # Historical data seeding & vector indexing
│   ├── test_milestone3.py              # Consolidated Milestone 3 test suite
│   └── README.md                       # Milestone 3 technical documentation
│
├── Milestone_4/                        # Enterprise Dashboard, Cloud Sync & Production Deployment
│   ├── api.py                          # Enterprise FastAPI REST API with Auth, Cloud Sync, Reports
│   ├── app.py                          # 7-Tab Streamlit Dashboard with Plotly visual analytics
│   ├── config.py                       # Production configuration & environment management
│   ├── models.py                       # SQLAlchemy models (Users, Meetings, ActionItems, Syncs, AuditLogs)
│   ├── schemas.py                      # Pydantic models for Auth, Search, RAG, Integrations, Reports
│   ├── database.py                     # Relational engine with multi-tenant access control
│   ├── vector_store.py                 # Persistent ChromaDB vector integration
│   ├── embedding_service.py            # Dynamic multi-entity embeddings
│   ├── search_service.py               # Sub-3s SLA semantic search with score tracking
│   ├── rag_service.py                  # Grounded Gemini RAG pipeline with attribution cards
│   ├── zoom_service.py                 # Zoom Cloud Recording ingestion & duplicate prevention
│   ├── google_meet_service.py          # Google Meet Drive ingestion & duplicate prevention
│   ├── report_service.py               # ReportLab PDF & RFC-4180 CSV export generation
│   ├── analytics_service.py            # Workload analytics & metric computation
│   ├── seed_data.py                    # Enterprise seeding & ChromaDB indexing
│   ├── Dockerfile                      # Multi-stage production container build
│   ├── docker-compose.yml              # Container orchestration with volume persistence
│   ├── run_all.bat                     # Windows one-click dual-service launcher
│   ├── run_backend.bat                 # FastAPI backend launcher
│   ├── run_frontend.bat                # Streamlit UI launcher
│   ├── test_milestone4.py              # Master test suite verifying all 10 tasks
│   ├── test_task_6_reports.py          # PDF/CSV report generation verification
│   ├── test_task_7_security.py         # Multi-tenant isolation & 403 test suite
│   ├── test_tasks_4_to_5_integrations.py # Zoom & Google Meet sync test suite
│   ├── test_task_8_end_to_end.py       # Autonomous 14-step workflow test suite
│   ├── test_task_9_performance_and_reliability.py # SLA latency & stress tests
│   ├── ARCHITECTURE.md                 # Deep architectural specification
│   └── README.md                       # Milestone 4 comprehensive documentation
│
├── packages.txt                        # System dependencies for Streamlit Cloud
├── LICENSE                             # MIT License
└── README.md                           # Main repository landing page & roadmap
```

---

## 🤝 Contributing & License

Contributions, issues, and feature suggestions are welcome!

1. Fork the repository
2. Create your branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

---

<div align="center">

Built with ❤️ by **[Nipun Bhadane](https://github.com/Nipunbhadane123)**

*Empowering teams with autonomous meeting intelligence.*

</div>
