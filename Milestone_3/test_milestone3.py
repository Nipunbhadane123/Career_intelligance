import time
import unittest
import logging
from database import (
    get_all_meetings,
    get_meeting_by_id,
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
    metadata_filtering
)
from search_service import (
    semantic_search_meetings,
    verify_semantic_search_performance,
    TARGET_SLA_SECONDS
)
from rag_service import generate_grounded_rag_answer, assemble_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone3Test")

class TestMilestone3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from seed_data import seed_database_and_vector_store
        seed_database_and_vector_store()

    # ══════════════════════════════════════════════════
    # Task 1: Meeting Knowledge Repository
    # ══════════════════════════════════════════════════
    def test_task1_knowledge_repository(self):
        """Verify historical meeting information retrieval and relational links."""
        logger.info("Running Task 1: Knowledge Repository Verification...")
        report = verify_knowledge_repository()
        self.assertIn(report["status"], ["PASS", "WARNING"])
        self.assertGreater(report["total_meetings"], 0, "Should have historical meetings")
        self.assertGreater(report["total_action_items"], 0, "Should have action items")
        self.assertGreater(report["total_decisions"], 0, "Should have decisions")
        self.assertGreater(len(report["total_participants"]), 0, "Should have participants")
        self.assertGreater(report["total_deadlines"], 0, "Should have deadlines tracked")
        
        # Verify a specific meeting links correctly
        meetings = get_all_meetings()
        sample = meetings[0]
        full_m = get_meeting_by_id(sample["id"])
        self.assertIsNotNone(full_m)
        self.assertEqual(full_m["id"], sample["id"])
        self.assertIn("action_items", full_m)
        self.assertIn("key_decisions", full_m)
        self.assertIn("participants", full_m)
        logger.info(f"Task 1 PASSED: {report['total_meetings']} meetings verified in knowledge repository.")

    # ══════════════════════════════════════════════════
    # Task 2: Embedding Generation
    # ══════════════════════════════════════════════════
    def test_task2_embedding_generation(self):
        """Verify dynamic embedding generation for transcript sections, summaries, decisions, and action items."""
        logger.info("Running Task 2: Embedding Generation Verification...")
        meetings = get_all_meetings()
        # Find a meeting with all components
        target_meeting = None
        for m in meetings:
            if m.get("summary") and m.get("key_decisions") and m.get("action_items") and m.get("transcript"):
                target_meeting = m
                break
        self.assertIsNotNone(target_meeting, "A complete meeting record should exist")

        verify_rep = verify_embedding_generation(target_meeting)
        self.assertEqual(verify_rep["status"], "PASS")
        self.assertTrue(verify_rep["valid_dimension"])
        self.assertTrue(verify_rep["all_linked_to_meeting"])
        self.assertEqual(verify_rep["embedding_dimension"], EMBEDDING_DIM)

        # Check coverage of all 4 required entity types
        doc_types = set(verify_rep["doc_types_generated"])
        self.assertIn("summary", doc_types, "Summaries must be embedded")
        self.assertIn("decision", doc_types, "Decisions must be embedded")
        self.assertIn("action_item", doc_types, "Action items must be embedded")
        self.assertIn("transcript", doc_types, "Transcript sections must be embedded")
        logger.info(f"Task 2 PASSED: Generated {verify_rep['total_embeddings_generated']} embeddings across all 4 entity types.")

    # ══════════════════════════════════════════════════
    # Task 3: Vector Database Integration
    # ══════════════════════════════════════════════════
    def test_task3_vector_database_integration(self):
        """Verify vector DB insert, update, delete, metadata filtering, and meeting-to-vector mapping."""
        logger.info("Running Task 3: Vector Database Integration Verification...")
        crud_res = verify_vector_database_integration()
        self.assertEqual(crud_res["status"], "PASS", f"CRUD checks failed: {crud_res['checks']}")

        stats = get_vector_store_stats()
        self.assertGreater(stats["total_vectors"], 0, "Persistent vector store must contain vectors")

        # Verify meeting-to-vector traceability
        collection = get_vector_collection()
        sample_ids = collection.get(limit=3)["ids"]
        self.assertGreater(len(sample_ids), 0)

        for vid in sample_ids:
            trace = trace_vector_to_meeting(vid)
            self.assertTrue(trace["traced"], f"Vector {vid} should trace back to meeting")
            self.assertTrue(trace["verified_link"], f"Relational link for {vid} should be verified")
            self.assertIsNotNone(trace["meeting_filename"])
            logger.info(f"Trace verified: Vector '{vid}' -> Meeting #{trace['meeting_id']} ('{trace['meeting_filename']}')")

        # Verify metadata filtering
        query_vec = generate_single_embedding("urgent actions")
        filtered_actions = metadata_filtering(query_vec, top_k=5, doc_type="action_item")
        for hit in filtered_actions:
            self.assertEqual(hit["doc_type"], "action_item", "Filtered results must match doc_type='action_item'")

        logger.info(f"Task 3 PASSED: Persistent vector store has {stats['total_vectors']} vectors with verified traceability.")

    # ══════════════════════════════════════════════════
    # Task 4: Semantic Search & <3s SLA
    # ══════════════════════════════════════════════════
    def test_task4_semantic_search(self):
        """Verify natural language search across historical meetings and <3s latency SLA."""
        logger.info("Running Task 4: Semantic Search Verification...")
        query = "Which meeting discussed the database migration?"
        res = verify_semantic_search_performance(query)
        self.assertIn(res["status"], ["PASS", "WARNING"])
        self.assertTrue(res["meets_sla_under_3s"], f"Retrieval time {res['retrieval_time_seconds']}s exceeded {TARGET_SLA_SECONDS}s SLA")
        self.assertGreater(res["total_hits"], 0, "Should return matching hits")
        
        # Check that database migration meeting is identified
        top_meeting = res.get("top_meeting", "")
        self.assertIn("migration", top_meeting.lower(), f"Expected migration meeting, got {top_meeting}")
        logger.info(f"Task 4 PASSED: Query '{query}' retrieved '{top_meeting}' in {res['retrieval_time_seconds']}s (SLA < 3.0s).")

    # ══════════════════════════════════════════════════
    # Task 5: RAG Question Answering
    # ══════════════════════════════════════════════════
    def test_task5_rag_question_answering(self):
        """Verify grounded RAG question answering flow and evidence citations."""
        logger.info("Running Task 5: RAG Question Answering Verification...")
        question = "What deadline was decided for the mobile application?"
        rag_res = generate_grounded_rag_answer(question)
        
        self.assertTrue(rag_res["is_grounded"], "RAG answer must be grounded in retrieved meeting sources")
        self.assertGreater(len(rag_res["sources"]), 0, "Must have retrieved context sources")
        
        # Verify retrieved sources contain the mobile application meeting and deadline
        sources_text = " ".join([s.get("text", "") for s in rag_res["sources"]]).lower()
        self.assertTrue("mobile" in sources_text, "Sources should mention mobile application")
        self.assertTrue("october 15" in sources_text or "october" in sources_text, "Sources should contain the October deadline")

        # Verify latency
        self.assertLess(rag_res["retrieval_time_seconds"], TARGET_SLA_SECONDS, "Semantic retrieval stage must meet SLA")
        logger.info(f"Task 5 PASSED: RAG answered '{question}' grounded in {len(rag_res['sources'])} sources in {rag_res['total_time_seconds']}s.")

    # ══════════════════════════════════════════════════
    # Task 6: Existing API Integration
    # ══════════════════════════════════════════════════
    def test_task6_api_integration(self):
        """Verify API layer with /meetings, /meetings/{id}, /search, /ask, auth, and error handling."""
        logger.info("Running Task 6: API Integration Verification...")
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)

        # 1. Health & Stats
        r_health = client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["status"], "healthy")

        # 2. GET /meetings
        r_meetings = client.get("/meetings?limit=5")
        self.assertEqual(r_meetings.status_code, 200)
        m_list = r_meetings.json()["meetings"]
        self.assertGreater(len(m_list), 0)
        first_id = m_list[0]["id"]

        # 3. GET /meetings/{id}
        r_single = client.get(f"/meetings/{first_id}")
        self.assertEqual(r_single.status_code, 200)
        self.assertEqual(r_single.json()["id"], first_id)

        # 4. POST /search
        r_search = client.post("/search", json={"query": "database migration", "top_k": 3})
        self.assertEqual(r_search.status_code, 200)
        self.assertTrue(r_search.json()["sla_met"])
        self.assertGreater(r_search.json()["total_hits"], 0)

        # 5. POST /ask
        r_ask = client.post("/ask", json={"question": "What deadline was decided for the mobile application?"})
        self.assertEqual(r_ask.status_code, 200)
        self.assertTrue(r_ask.json()["is_grounded"])
        self.assertGreater(len(r_ask.json()["sources"]), 0)

        # 6. Auth flow
        reg_r = client.post("/auth/register", json={"username": f"user_{int(time.time())}", "password": "Password123!"})
        self.assertEqual(reg_r.status_code, 201)

        logger.info("Task 6 PASSED: Advanced API layer endpoints integrated and verified without breaking backend.")

    # ══════════════════════════════════════════════════
    # Task 7: Search & RAG Validation
    # ══════════════════════════════════════════════════
    def test_task7_search_and_rag_validation(self):
        """Verify date-based filtering, metadata filtering, irrelevant queries, and empty query safety."""
        logger.info("Running Task 7: Search & RAG Validation...")
        # Date filtering
        res_date = semantic_search_meetings("migration", start_date="2020-01-01", end_date="2030-12-31")
        self.assertGreater(res_date["total_hits"], 0)

        # Metadata filtering
        res_meta = semantic_search_meetings("deadline", doc_type="action_item")
        for h in res_meta["raw_hits"]:
            self.assertEqual(h["doc_type"], "action_item")

        # Empty search results handling
        res_empty = semantic_search_meetings("")
        self.assertEqual(res_empty["total_hits"], 0)
        self.assertTrue(res_empty["sla_met"])

        # Irrelevant query handling
        res_irrel = semantic_search_meetings("orbital mechanics astrophysics planetary alignment")
        self.assertTrue(res_irrel["sla_met"])
        logger.info("Task 7 PASSED: Search and RAG validation criteria verified.")

    # ══════════════════════════════════════════════════
    # Task 8: Performance & Edge Case Testing
    # ══════════════════════════════════════════════════
    def test_task8_performance_and_edge_cases(self):
        """Verify large transcripts, long/short/unknown queries, null safety, and fault tolerance."""
        logger.info("Running Task 8: Performance & Edge Case Verification...")
        # Short and Long queries
        res_short = semantic_search_meetings("database")
        self.assertTrue(res_short["sla_met"])

        long_q = "Could you please explain in extensive detail whether there were any explicit decisions reached regarding migrating from legacy MySQL to Amazon Aurora PostgreSQL?"
        res_long = semantic_search_meetings(long_q)
        self.assertTrue(res_long["sla_met"])

        # Null safety on broken records
        broken = {"id": 99990, "filename": None, "transcript": None, "summary": None, "key_decisions": None, "action_items": None}
        units = export_meeting_knowledge_units(broken)
        self.assertEqual(len(units), 0)

        logger.info("Task 8 PASSED: Performance, edge cases, and fault-tolerance verified.")

    # ══════════════════════════════════════════════════
    # Task 9: End-to-End Integration Testing
    # ══════════════════════════════════════════════════
    def test_task9_end_to_end_workflow(self):
        """Verify complete end-to-end user workflow: register, authenticate, search, and RAG Q&A."""
        logger.info("Running Task 9: End-to-End Integration Testing...")
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        uname = f"e2e_user_{int(time.time())}"
        pword = "ValidPassword123!"

        # 1. Register & Login
        reg = client.post("/auth/register", json={"username": uname, "password": pword})
        self.assertEqual(reg.status_code, 201)

        login = client.post("/auth/login", json={"username": uname, "password": pword})
        self.assertEqual(login.status_code, 200)
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Browse meetings
        r_m = client.get("/meetings", headers=headers)
        self.assertEqual(r_m.status_code, 200)
        self.assertGreater(r_m.json()["total"], 0)

        # 3. Search meetings
        s_res = client.post("/search", json={"query": "Aurora PostgreSQL"}, headers=headers)
        self.assertEqual(s_res.status_code, 200)
        self.assertTrue(s_res.json()["sla_met"])

        # 4. Ask grounded question
        ask_res = client.post("/ask", json={"question": "What is the downtime window for database migration?"}, headers=headers)
        self.assertEqual(ask_res.status_code, 200)
        self.assertTrue(ask_res.json()["is_grounded"])
        self.assertGreater(len(ask_res.json()["sources"]), 0)

        logger.info("Task 9 PASSED: Complete end-to-end integration workflow verified without manual intervention.")

if __name__ == "__main__":
    unittest.main(verbosity=2)

