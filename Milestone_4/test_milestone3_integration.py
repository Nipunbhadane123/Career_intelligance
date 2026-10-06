import time
import unittest
import logging
from fastapi.testclient import TestClient
from api import app
from database import (
    init_db,
    get_all_meetings,
    get_meeting_by_id,
    get_user_meetings,
    verify_knowledge_repository,
    export_meeting_knowledge_units
)
from embedding_service import (
    generate_single_embedding,
    generate_meeting_embeddings,
    verify_embedding_generation,
    EMBEDDING_DIM
)
from vector_store import (
    get_vector_collection,
    get_vector_store_stats,
    verify_vector_database_integration,
    trace_vector_to_meeting,
    metadata_filtering,
    similarity_search
)
from search_service import (
    semantic_search_meetings,
    verify_semantic_search_performance,
    TARGET_SLA_SECONDS
)
from rag_service import (
    generate_grounded_rag_answer,
    stream_grounded_rag_answer,
    assemble_context
)
from seed_data import seed_database_and_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone3IntegrationInM4")

class TestMilestone3FeaturesInMilestone4(unittest.TestCase):
    """
    Verification Suite confirming that 100% of Milestone 3 features
    are fully functional and continuously integrated into Milestone 4.
    """

    @classmethod
    def setUpClass(cls):
        seed_database_and_vector_store()
        cls.client = TestClient(app)
        cls.ts = int(time.time() * 1000)
        cls.username = f"m3_verify_user_{cls.ts}"
        cls.password = "M3VerifyPassword123!"

        cls.client.post("/auth/register", json={"username": cls.username, "password": cls.password})
        r_login = cls.client.post("/auth/login", json={"username": cls.username, "password": cls.password})
        cls.token = r_login.json()["access_token"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    # ══════════════════════════════════════════════════
    # Task 1: Knowledge Repository
    # ══════════════════════════════════════════════════
    def test_task1_knowledge_repository(self):
        """Verify relational repository integrity, foreign keys, and entity graphs."""
        report = verify_knowledge_repository()
        self.assertIn(report["status"], ["PASS", "WARNING"])
        self.assertGreater(report["total_meetings"], 0)
        self.assertGreater(report["total_action_items"], 0)
        self.assertGreater(report["total_decisions"], 0)
        self.assertGreater(len(report["total_participants"]), 0)

        meetings = get_all_meetings()
        sample = meetings[0]
        full_m = get_meeting_by_id(sample["id"])
        self.assertIsNotNone(full_m)
        self.assertEqual(full_m["id"], sample["id"])
        self.assertIn("action_items", full_m)
        self.assertIn("key_decisions", full_m)
        self.assertIn("participants", full_m)
        logger.info(f"M3 Task 1 PASSED in M4: {report['total_meetings']} historical meetings verified.")

    # ══════════════════════════════════════════════════
    # Task 2: Multi-Entity Dynamic Embeddings
    # ══════════════════════════════════════════════════
    def test_task2_dynamic_embedding_generation(self):
        """Verify 384-dimensional dense embeddings generated across all 4 entity types."""
        meetings = get_all_meetings()
        complete_meeting = None
        for m in meetings:
            if m.get("summary") and m.get("key_decisions") and m.get("action_items") and m.get("transcript"):
                complete_meeting = m
                break
        self.assertIsNotNone(complete_meeting, "A meeting with all entity types must exist")

        verify_rep = verify_embedding_generation(complete_meeting)
        self.assertEqual(verify_rep["status"], "PASS")
        self.assertTrue(verify_rep["valid_dimension"])
        self.assertTrue(verify_rep["all_linked_to_meeting"])
        self.assertEqual(verify_rep["embedding_dimension"], 384)

        doc_types = set(verify_rep["doc_types_generated"])
        self.assertIn("summary", doc_types)
        self.assertIn("decision", doc_types)
        self.assertIn("action_item", doc_types)
        self.assertIn("transcript", doc_types)
        logger.info(f"M3 Task 2 PASSED in M4: Verified all 4 entity types with dimension {EMBEDDING_DIM}.")

    # ══════════════════════════════════════════════════
    # Task 3: Vector Database Integration & Traceability
    # ══════════════════════════════════════════════════
    def test_task3_vector_database_and_traceability(self):
        """Verify ChromaDB CRUD, metadata filtering, and meeting-to-vector traceability."""
        crud_res = verify_vector_database_integration()
        self.assertEqual(crud_res["status"], "PASS")

        stats = get_vector_store_stats()
        self.assertGreater(stats["total_vectors"], 0)
        self.assertGreater(stats["unique_meetings_indexed"], 0)
        self.assertIn("breakdown", stats)

        # Trace vector back to relational meeting
        coll = get_vector_collection()
        sample_ids = coll.get(limit=3)["ids"]
        self.assertGreater(len(sample_ids), 0)

        for vid in sample_ids:
            trace = trace_vector_to_meeting(vid)
            self.assertTrue(trace["traced"])
            self.assertTrue(trace["verified_link"])
            self.assertIsNotNone(trace["meeting_filename"])
            self.assertIsNotNone(trace["meeting"])

        # Metadata filtering
        q_emb = generate_single_embedding("security updates and permissions")
        hits = metadata_filtering(q_emb, top_k=5, doc_type="action_item")
        for h in hits:
            self.assertEqual(h["doc_type"], "action_item")
        logger.info("M3 Task 3 PASSED in M4: Persistent ChromaDB CRUD & bidirectional traceability verified.")

    # ══════════════════════════════════════════════════
    # Task 4: Sub-3s Semantic Search
    # ══════════════════════════════════════════════════
    def test_task4_sub3s_semantic_search(self):
        """Verify natural language search across meetings under 3.0s SLA requirement."""
        query = "Which meeting discussed the database migration?"
        res = verify_semantic_search_performance(query)
        self.assertIn(res["status"], ["PASS", "WARNING"])
        self.assertTrue(res["meets_sla_under_3s"])
        self.assertGreater(res["total_hits"], 0)
        top_m = res.get("top_meeting", "").lower()
        self.assertIn("migration", top_m)

        # Verify matched_meetings grouped return structure
        s_res = semantic_search_meetings(query, top_k=6)
        self.assertIn("matched_meetings", s_res)
        self.assertGreater(len(s_res["matched_meetings"]), 0)
        first_group = s_res["matched_meetings"][0]
        self.assertIn("filename", first_group)
        self.assertIn("best_score", first_group)
        self.assertIn("matching_entities", first_group)
        logger.info(f"M3 Task 4 PASSED in M4: Retrieved in {res['retrieval_time_seconds']}s (SLA < 3.0s).")

    # ══════════════════════════════════════════════════
    # Task 5: Grounded RAG Q&A & Streaming
    # ══════════════════════════════════════════════════
    def test_task5_grounded_rag_and_streaming(self):
        """Verify grounded answers, source citations, and streaming generator."""
        question = "What deadline was decided for the mobile application?"
        rag_res = generate_grounded_rag_answer(question, top_k=5)
        self.assertTrue(rag_res["is_grounded"])
        self.assertGreater(len(rag_res["answer"]), 0)
        self.assertGreater(len(rag_res["sources"]), 0)
        self.assertLess(rag_res["retrieval_time_seconds"], TARGET_SLA_SECONDS)

        # Test streaming generator interface
        stream_chunks = []
        for chunk, sources, s_res in stream_grounded_rag_answer(question, top_k=3):
            stream_chunks.append(chunk)
            if sources:
                self.assertGreater(len(sources), 0)

        self.assertGreater(len(stream_chunks), 0)
        streamed_text = "".join(stream_chunks)
        self.assertGreater(len(streamed_text), 0)
        logger.info("M3 Task 5 PASSED in M4: Grounded RAG and streaming generator verified.")

    # ══════════════════════════════════════════════════
    # Task 6: REST API Integration
    # ══════════════════════════════════════════════════
    def test_task6_api_endpoints(self):
        """Verify /meetings, /meetings/{id}, /search, /ask, /health, /stats."""
        # /health
        r_h = self.client.get("/health")
        self.assertEqual(r_h.status_code, 200)

        # /stats
        r_s = self.client.get("/stats")
        self.assertEqual(r_s.status_code, 200)

        # /meetings
        r_m = self.client.get("/meetings?limit=5", headers=self.headers)
        self.assertEqual(r_m.status_code, 200)
        m_list = r_m.json()["meetings"]
        self.assertGreater(len(m_list), 0)
        target_id = m_list[0]["id"]

        # /meetings/{id}
        r_det = self.client.get(f"/meetings/{target_id}", headers=self.headers)
        self.assertEqual(r_det.status_code, 200)

        # POST /search
        r_search = self.client.post("/search", json={"query": "database migration", "top_k": 5}, headers=self.headers)
        self.assertEqual(r_search.status_code, 200)
        self.assertIn("results", r_search.json())

        # POST /ask
        r_ask = self.client.post("/ask", json={"question": "What is the release deadline?", "top_k": 3}, headers=self.headers)
        self.assertEqual(r_ask.status_code, 200)
        self.assertIn("answer", r_ask.json())
        logger.info("M3 Task 6 PASSED in M4: All Milestone 3 REST API endpoints active and responsive.")

    # ══════════════════════════════════════════════════
    # Task 7: Search & RAG Validation
    # ══════════════════════════════════════════════════
    def test_task7_search_and_rag_validation(self):
        """Verify date filtering, doc_type filtering, and empty query handling."""
        # 1. Date filter
        s_date = semantic_search_meetings("meeting", start_date="2026-01-01", end_date="2026-12-31")
        self.assertTrue(s_date["sla_met"])

        # 2. Doc type filter
        s_doc = semantic_search_meetings("decision", doc_type="decision", top_k=4)
        for hit in s_doc["results"]:
            self.assertEqual(hit["doc_type"], "decision")

        # 3. Empty query safety
        s_empty = semantic_search_meetings("", top_k=5)
        self.assertEqual(s_empty["total_results"], 0)
        self.assertTrue(s_empty["sla_met"])
        logger.info("M3 Task 7 PASSED in M4: Search & RAG validation constraints confirmed.")

    # ══════════════════════════════════════════════════
    # Task 8: Performance, Edge Cases & Fault Tolerance
    # ══════════════════════════════════════════════════
    def test_task8_edge_cases_and_fault_tolerance(self):
        """Verify large inputs, unknown questions, null safety, and fault tolerance."""
        # 1. Unknown / nonsensical query
        res_unk = generate_grounded_rag_answer("What is the recipe for planetary quantum muffins in outer space?", top_k=3)
        self.assertIn("answer", res_unk)

        # 2. Long question handling
        long_q = "Can you describe in extensive detail all architectural considerations, tradeoffs, deadlines, rollback protocols, database schema migrations, and speaker comments mentioned in the technical syncs?"
        res_long = generate_grounded_rag_answer(long_q, top_k=4)
        self.assertIn("answer", res_long)
        self.assertGreater(len(res_long["answer"]), 0)
        logger.info("M3 Task 8 PASSED in M4: Edge cases, long queries, and unknown questions handled safely.")

    # ══════════════════════════════════════════════════
    # Task 9: Continuous Integration Unified Flow
    # ══════════════════════════════════════════════════
    def test_task9_continuous_integration_flow(self):
        """
        Verify complete seamless workflow:
        User Access -> Semantic Search (M3) -> Grounded RAG (M3) -> Details & Analytics (M4) -> PDF Export (M4).
        """
        # Step 1: Semantic Search
        s_res = self.client.post("/search", json={"query": "database migration", "top_k": 3}, headers=self.headers)
        self.assertEqual(s_res.status_code, 200)
        m_id = s_res.json()["results"][0]["meeting_id"]

        # Step 2: Grounded RAG
        q_res = self.client.post("/ask", json={"question": "What database was chosen?", "meeting_id": m_id}, headers=self.headers)
        self.assertEqual(q_res.status_code, 200)
        self.assertTrue(q_res.json()["is_grounded"])

        # Step 3: Meeting Analytics (M4 feature)
        a_res = self.client.get(f"/analytics/meeting/{m_id}", headers=self.headers)
        self.assertEqual(a_res.status_code, 200)

        # Step 4: PDF Export (M4 feature)
        pdf_res = self.client.get(f"/reports/pdf/{m_id}", headers=self.headers)
        self.assertEqual(pdf_res.status_code, 200)
        self.assertTrue(pdf_res.content.startswith(b"%PDF"))
        logger.info("M3 Task 9 PASSED in M4: Unified M3 + M4 continuous pipeline validated 100%.")

if __name__ == "__main__":
    unittest.main()
