# Milestone 3 — Meeting Knowledge Repository & Semantic RAG

**Milestone 3** elevates the AI Meeting Synthesizer platform into an enterprise-grade, persistent meeting knowledge base. Historical meetings are structured across relational entities and synchronized into a high-performance vector database, enabling sub-3-second natural language search, strictly grounded RAG question answering with source citations, and an advanced REST API layer.

---

## 🚀 Key Milestone 3 Features

| Task | Feature | Description | Status |
|---|---|---|---|
| **Task 1** | **Meeting Knowledge Repository** | Organizes historical meeting metadata, transcripts, summaries, decisions, action items, participants, and deadlines into a unified relational knowledge graph. | ✅ Complete |
| **Task 2** | **Multi-Entity Dynamic Embedding Generation** | Dynamically embeds meeting entities (summaries, decisions, action items, transcript sections) using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions). | ✅ Complete |
| **Task 3** | **Persistent Vector Database Integration** | Full ChromaDB CRUD operations (Insert, Update, Delete), cosine similarity search, metadata filtering (`doc_type`, `meeting_id`), and verified meeting-to-vector traceability. | ✅ Complete |
| **Task 4** | **Sub-3s Semantic Search** | Natural-language query search across historical meetings (`"Which meeting discussed the database migration?"`) with dynamic ranking and sub-3-second retrieval SLA tracking (~0.06s typical). | ✅ Complete |
| **Task 5** | **Grounded RAG Question Answering** | Gemini-powered Q&A pipeline answering user questions (`"What deadline was decided for the mobile application?"`) grounded strictly in retrieved meeting records with expandable evidence citations. | ✅ Complete |
| **Task 6** | **Existing API Integration** | FastAPI REST API exposing `/meetings`, `/meetings/{id}`, `/search`, and `/ask` with Bearer auth, standardized error handling, and structured request logging. | ✅ Complete |
| **Task 7** | **Search & RAG Validation** | Verification of retrieval relevance, irrelevant query handling, multiple matching meetings, date-based filtering, doc_type filtering, citation accuracy, and empty query safety. | ✅ Complete |
| **Task 8** | **Performance & Edge Case Testing** | Stress and fault-tolerance testing: large transcripts (120+ lines), long/short/unknown queries, null safety on missing fields, vector DB fault tolerance, and LLM timeout fallbacks. | ✅ Complete |
| **Task 9** | **End-to-End Integration Testing** | Automated end-to-end integration test client verifying registration, login, meeting browsing, semantic search, and RAG Q&A with zero manual intervention. | ✅ Complete |
| **Task 10** | **Project Cleanup & Final Validation** | Code refactoring, duplicate removal, fast local-cache embedding loading, full test suite consolidation, and comprehensive API documentation. | ✅ Complete |

---

## 🔄 System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Storage ["Repository Layer"]
        A[Relational DB: SQLite] -->|SQLAlchemy| B[Meeting Knowledge Units]
        B -->|Metadata, Summaries, Decisions, Action Items, Transcripts| C[Dynamic Embedder: all-MiniLM-L6-v2]
        C -->|384-d Vectors| D[(ChromaDB Persistent Store)]
    end

    subgraph SearchFlow ["Task 4: Sub-3s Semantic Search"]
        E[User Query: 'Which meeting discussed database migration?'] --> F[Query Embedding Generator]
        F --> G[Vector Similarity Search & Metadata Filter]
        D --> G
        G --> H[Meeting-Level Result Aggregator & Ranker]
        H --> I[Dynamic Search Results with <3s SLA Badge]
    end

    subgraph RAGFlow ["Task 5: Grounded RAG Q&A"]
        J[User Question: 'What deadline was decided for mobile app?'] --> K[Semantic Retrieval]
        D --> K
        K --> L[Attributed Context Assembly]
        L --> M[Google Gemini 2.5 Flash]
        M --> N[Grounded Answer + Evidence Source Cards]
    end

    subgraph APILayer ["Task 6: Advanced API Layer"]
        O[Client Apps / Frontend] --> P[FastAPI Application]
        P -->|Auth: /auth/login, /auth/register| A
        P -->|Meetings: /meetings, /meetings/{id}| A
        P -->|Search: /search| H
        P -->|Ask: /ask| N
    end
```

---

## 📡 REST API Reference

The production API is powered by **FastAPI** in [`api.py`](api.py). Access the interactive OpenAPI Swagger UI at `http://localhost:8000/docs`.

### Authentication Endpoints
- `POST /auth/register`: Create user account (`username`, `password`).
- `POST /auth/login`: Authenticate and receive a Bearer access token.

### Meeting Intelligence Endpoints
- `GET /meetings`: List historical meetings with pagination (`limit`, `offset`), `user_id` filtering, and date range filters (`start_date`, `end_date`).
- `GET /meetings/{id}`: Retrieve complete relational graph for a meeting (transcripts, summary, action items with assignees & deadlines, key decisions, participants).

### Semantic Search & RAG Endpoints
- `POST /search` & `GET /search`: Natural language vector search across historical meetings with sub-3-second SLA tracking.
  - Query parameters / JSON payload: `query` (required), `top_k` (default 8), `doc_type` (`summary`, `decision`, `action_item`, `transcript`), `meeting_id`, `start_date`, `end_date`.
- `POST /ask`: Grounded question answering pipeline.
  - Payload: `{"question": "What deadline was decided for the mobile application?", "top_k": 5}`.
  - Response: Grounded synthesis, list of attributed meeting sources, and SLA retrieval time.

### Monitoring & System Endpoints
- `GET /health`: Health check verifying relational database and vector store connectivity.
- `GET /stats`: Real-time vector store breakdown and knowledge repository counts.

---

## 📁 Project Structure

```text
Milestone_3/
├── api.py                            # FastAPI production REST API (/meetings, /search, /ask, auth)
├── app.py                            # Streamlit application with Knowledge Search & AI Q&A UI
├── database.py                       # Relational database engine, date filtering, knowledge units
├── models.py                         # SQLAlchemy ORM models (Meeting, ActionItem, KeyDecision, etc.)
├── schemas.py                        # Pydantic request/response validation schemas
├── embedding_service.py              # Multi-entity dynamic embedding generator (all-MiniLM-L6-v2)
├── vector_store.py                   # Persistent ChromaDB integration, CRUD, date & metadata filters
├── search_service.py                 # Semantic search engine with sub-3s SLA timing
├── rag_service.py                    # Grounded RAG workflow, prompt grounding, streaming generator
├── seed_data.py                      # Database migration and vector store indexing script
├── test_milestone3.py                # Consolidated master test suite covering Tasks 1 - 9
├── test_api_integration.py          # End-to-end API test suite (TestClient)
├── test_validation_and_edge_cases.py # Comprehensive validation & edge-case test suite (Tasks 7 & 8)
├── requirements.txt                  # Python dependencies (FastAPI, Uvicorn, ChromaDB, etc.)
├── meeting_intelligence.db           # SQLite database storing historical meetings
└── chroma_db/                        # Persistent ChromaDB vector index
```

---

## 🧪 Verification & Automated Tests

All 3 automated test suites are validated and passing:

### 1. Master Milestone 3 Test Suite (Tasks 1 – 9)
```bash
python test_milestone3.py
```
- `test_task1_knowledge_repository`: ✅ **PASSED** (10 historical meetings verified with relational integrity).
- `test_task2_embedding_generation`: ✅ **PASSED** (384-dimensional dense vectors generated across all 4 entity types).
- `test_task3_vector_database_integration`: ✅ **PASSED** (CRUD insert, update, delete, metadata filtering, and meeting-to-vector mapping verified).
- `test_task4_semantic_search`: ✅ **PASSED** (Retrieved database migration meeting in **0.06s**, well under 3.0s SLA).
- `test_task5_rag_question_answering`: ✅ **PASSED** (Grounded answer citing October 15, 2026 deadline for mobile application).
- `test_task6_api_integration`: ✅ **PASSED** (FastAPI endpoints `/health`, `/meetings`, `/meetings/{id}`, `/search`, `/ask` verified).
- `test_task7_search_and_rag_validation`: ✅ **PASSED** (Verified date filtering, doc_type filtering, irrelevant queries, and empty query safety).
- `test_task8_performance_and_edge_cases`: ✅ **PASSED** (Verified long/short queries, null safety on missing fields, and fault tolerance).
- `test_task9_end_to_end_workflow`: ✅ **PASSED** (Full end-to-end user workflow: register, authenticate, search, and RAG Q&A verified).

### 2. Search & RAG Validation and Edge Cases (Tasks 7 & 8)
```bash
python test_validation_and_edge_cases.py
```
- 16 automated tests covering large transcripts, short/long/unknown queries, null safety, vector DB error simulation, and LLM failure fallback.

### 3. End-to-End API Integration Suite (Tasks 6 & 9)
```bash
python test_api_integration.py
```
- 9 automated tests verifying HTTP status codes (200, 201, 400, 401, 404, 422), request timing, pagination, and SLA compliance.

---

## 💻 Running the Services

### 1. Launch the REST API Server (Uvicorn)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000/docs** in your browser for the Swagger documentation.

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Navigate to **"Knowledge Search & AI Q&A"** to interact visually with semantic search and cross-meeting RAG chat.
