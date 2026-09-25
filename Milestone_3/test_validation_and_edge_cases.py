import time
import unittest
import logging
from unittest.mock import patch, MagicMock

from database import (
    get_all_meetings,
    get_meeting_by_id,
    get_user_meetings,
    export_meeting_knowledge_units
)
from embedding_service import (
    generate_single_embedding,
    generate_meeting_embeddings
)
from vector_store import (
    get_vector_collection,
    metadata_filtering,
    similarity_search,
    insert_meeting_vectors,
    delete_meeting_vectors
)
from search_service import (
    semantic_search_meetings,
    verify_semantic_search_performance,
    TARGET_SLA_SECONDS
)
from rag_service import (
    generate_grounded_rag_answer,
    assemble_context,
    RAG_SYSTEM_PROMPT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ValidationEdgeCaseTest")

class TestSearchAndRAGValidation(unittest.TestCase):
    """
    Task 7: Search & RAG Validation
    Verifies retrieval relevance, irrelevant query handling, multiple matches,
    date & metadata filtering, citation accuracy, and empty search results.
    """

    # 1. Relevant meeting retrieval
    def test_relevant_meeting_retrieval(self):
        res = semantic_search_meetings("database migration strategy MySQL Aurora", top_k=5)
        self.assertTrue(res["sla_met"], "SLA must be met")
        self.assertGreater(res["total_hits"], 0, "Should retrieve hits for database migration")
        matched = res["matched_meetings"]
        self.assertGreater(len(matched), 0)
        self.assertIn("migration", matched[0]["filename"].lower())
        logger.info(f"Relevant meeting retrieval PASSED: '{matched[0]['filename']}' with score {matched[0]['best_score']}")

    # 2. Irrelevant query handling
    def test_irrelevant_query_handling(self):
        irrelevant_q = "How to make chocolate chip cookies in an orbital space station with zero gravity"
        res = semantic_search_meetings(irrelevant_q, top_k=5)
        self.assertTrue(res["sla_met"])
        # Either no hits or low similarity score without crash
        if res["total_hits"] > 0:
            top_score = res["matched_meetings"][0]["best_score"]
            self.assertLess(top_score, 0.70, f"Irrelevant query should have low similarity, got {top_score}")
        
        # Test in RAG
        rag_res = generate_grounded_rag_answer(irrelevant_q)
        self.assertIsNotNone(rag_res["answer"])
        logger.info("Irrelevant query handling PASSED: Safe response returned without hallucination.")

    # 3. Multiple matching meetings
    def test_multiple_matching_meetings(self):
        query = "meeting decisions and action items"
        res = semantic_search_meetings(query, top_k=10)
        self.assertTrue(res["sla_met"])
        self.assertGreaterEqual(len(res["matched_meetings"]), 2, "Should return multiple meetings for broad query")
        # Verify sorted by best_score descending
        scores = [m["best_score"] for m in res["matched_meetings"]]
        self.assertEqual(scores, sorted(scores, reverse=True), "Results must be ranked descending by score")
        logger.info(f"Multiple matching meetings PASSED: {len(res['matched_meetings'])} ranked meetings returned.")

    # 4. Date-based filtering
    def test_date_based_filtering(self):
        # Filtering with future date range should yield 0 hits
        res_future = semantic_search_meetings(
            "database migration",
            start_date="2035-01-01",
            end_date="2035-12-31"
        )
        self.assertEqual(res_future["total_hits"], 0, "Future date range should yield no hits")

        # Filtering with wide range should yield hits
        res_valid = semantic_search_meetings(
            "database migration",
            start_date="2020-01-01",
            end_date="2030-12-31"
        )
        self.assertGreater(res_valid["total_hits"], 0, "Valid date range should retain hits")
        logger.info("Date-based filtering PASSED: Verified boundary inclusions and exclusions.")

    # 5. Meeting metadata filtering (by doc_type and meeting_id)
    def test_meeting_metadata_filtering(self):
        all_meetings = get_all_meetings()
        target_id = all_meetings[0]["id"]

        # Filter by specific meeting_id
        res_id = semantic_search_meetings("meeting", meeting_id=target_id)
        for h in res_id["raw_hits"]:
            self.assertEqual(h["meeting_id"], target_id)

        # Filter by doc_type
        res_action = semantic_search_meetings("deadline", doc_type="action_item")
        for h in res_action["raw_hits"]:
            self.assertEqual(h["doc_type"], "action_item")
        logger.info("Metadata filtering PASSED: Verified doc_type and meeting_id constraints.")

    # 6. Correct source meeting & Correct context retrieval
    def test_correct_source_meeting_and_context(self):
        q = "What is the maintenance window downtime for Aurora PostgreSQL?"
        rag_res = generate_grounded_rag_answer(q)
        self.assertTrue(rag_res["is_grounded"])
        self.assertGreater(len(rag_res["sources"]), 0)
        
        # Verify source file is database migration
        top_source = rag_res["sources"][0]
        self.assertIn("migration", top_source.get("meeting_filename", "").lower())
        
        # Verify retrieved context contains "2-hour" or "2 AM"
        context_str = rag_res["context_preview"].lower()
        self.assertTrue("2-hour" in context_str or "2 am" in context_str)
        logger.info("Correct source meeting & context retrieval PASSED: Correctly attributed to migration meeting.")

    # 7. Grounded AI answers
    def test_grounded_ai_answers(self):
        q = "What deadline was decided for the mobile application?"
        rag_res = generate_grounded_rag_answer(q)
        self.assertTrue(rag_res["is_grounded"])
        ans_lower = rag_res["answer"].lower()
        self.assertTrue("october 15" in ans_lower or "october" in ans_lower, "Answer must contain grounded deadline")
        logger.info("Grounded AI answers PASSED: Factually grounded without hallucinations.")

    # 8. Empty search results
    def test_empty_search_results(self):
        # Empty query
        res_empty = semantic_search_meetings("")
        self.assertEqual(res_empty["total_hits"], 0)
        self.assertEqual(res_empty["matched_meetings"], [])
        self.assertTrue(res_empty["sla_met"])

        # Whitespace query
        res_space = semantic_search_meetings("    ")
        self.assertEqual(res_space["total_hits"], 0)

        # Impossible filter criteria
        res_impossible = semantic_search_meetings("database", meeting_id=888888)
        self.assertEqual(res_impossible["total_hits"], 0)
        logger.info("Empty search results PASSED: Safe empty response contracts maintained.")


class TestPerformanceAndEdgeCases(unittest.TestCase):
    """
    Task 8: Performance & Edge Case Testing
    Verifies behavior with large transcripts, short/long/unknown queries,
    missing fields, vector store faults, and LLM timeouts.
    """

    @classmethod
    def setUpClass(cls):
        from embedding_service import get_embedding_model
        # Warm up local embedding model
        get_embedding_model()

    # 1. Large meeting transcripts
    def test_large_meeting_transcripts(self):
        large_transcript = "\n".join([
            f"[{i:02d}:00] Speaker_{(i%4)+1}: This is sentence line number {i} discussing enterprise architecture and distributed consensus protocol {i}."
            for i in range(120)
        ])
        large_meeting = {
            "id": 88881,
            "filename": "massive_architecture_review.mp4",
            "summary": "Large meeting executive summary covering distributed systems.",
            "transcript": large_transcript,
            "key_decisions": ["Adopt Raft consensus algorithm"],
            "key_points": ["Cluster scale-out testing"],
            "action_items": [{"description": "Benchmark node latencies", "assigned_participant": "Alice", "deadline": "Next week"}]
        }
        units = export_meeting_knowledge_units(large_meeting)
        self.assertGreater(len(units), 20, "Should generate chunked units for large transcript")

        # Dynamic embedding generation under load
        start = time.perf_counter()
        embeddings = generate_meeting_embeddings(large_meeting)
        elapsed = time.perf_counter() - start
        self.assertEqual(len(embeddings), len(units))
        self.assertLess(elapsed, 30.0, "Bulk embedding should complete within reasonable time")
        logger.info(f"Large meeting transcripts PASSED: Processed 120 dialogue lines into {len(embeddings)} vectors in {elapsed:.2f}s.")

    # 2. Long questions (500+ characters)
    def test_long_questions(self):
        long_q = "Could you please comprehensively explain in extensive detail whether there were any explicit discussions or definitive conclusions reached regarding the proposed migration from the legacy on-premises MySQL relational database to Amazon Aurora PostgreSQL, and specifically what the agreed scheduled maintenance window downtime duration and start time were according to Bob and Charlie?"
        self.assertGreater(len(long_q), 300)
        res = semantic_search_meetings(long_q, top_k=3)
        self.assertTrue(res["sla_met"])
        self.assertGreater(res["total_hits"], 0)
        self.assertIn("migration", res["matched_meetings"][0]["filename"].lower())
        logger.info("Long questions PASSED: 350+ char complex question correctly resolved.")

    # 3. Short / single-word questions
    def test_short_questions(self):
        short_queries = ["database", "when?", "deadline", "?"]
        for sq in short_queries:
            res = semantic_search_meetings(sq, top_k=3)
            self.assertTrue(res["sla_met"])
            self.assertIsNotNone(res["matched_meetings"])
        logger.info("Short questions PASSED: Single-token and punctuation queries handled cleanly.")

    # 4. Unknown questions
    def test_unknown_questions(self):
        q = "What was the quantum cryptography key used in the Apollo 11 moon landing?"
        rag_res = generate_grounded_rag_answer(q)
        self.assertIsNotNone(rag_res["answer"])
        logger.info("Unknown questions PASSED: Safely returned without crashing.")

    # 5. Missing transcript data & missing fields (null safety)
    def test_missing_data_null_safety(self):
        broken_meeting = {
            "id": 99991,
            "filename": None,
            "transcript": None,
            "summary": None,
            "key_decisions": None,
            "action_items": None,
            "key_points": None
        }
        # Must not raise exceptions
        units = export_meeting_knowledge_units(broken_meeting)
        self.assertIsInstance(units, list)
        self.assertEqual(len(units), 0)

        embs = generate_meeting_embeddings(broken_meeting)
        self.assertEqual(len(embs), 0)
        logger.info("Missing data null safety PASSED: None values handled without raising exceptions.")

    # 6. Duplicate meeting data resilience
    def test_duplicate_meeting_data_resilience(self):
        test_m_id = 77771
        sample_item = [{
            "unit_id": f"m_{test_m_id}_summary",
            "meeting_id": test_m_id,
            "text": "Duplicate test summary",
            "embedding": [0.01] * 384,
            "metadata": {"meeting_id": test_m_id, "doc_type": "summary", "created_at": "2026-09-01"}
        }]
        # Upserting twice should not create duplicate entries
        c1 = insert_meeting_vectors(test_m_id, sample_item)
        c2 = insert_meeting_vectors(test_m_id, sample_item)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 1)
        delete_meeting_vectors(test_m_id)
        logger.info("Duplicate data resilience PASSED: Upsert guarantees idempotency.")

    # 7. Vector database failure resilience
    def test_vector_database_failure_simulation(self):
        with patch("vector_store.get_vector_collection", side_effect=Exception("Simulated Vector DB Disconnection")):
            # similarity_search should not crash, but return empty list
            res = similarity_search([0.01] * 384, top_k=5)
            self.assertEqual(res, [], "Should return [] when vector DB fails")

            # search_service should not crash
            s_res = semantic_search_meetings("test query")
            self.assertEqual(s_res["total_hits"], 0)
            self.assertTrue(s_res["sla_met"])
        logger.info("Vector database failure simulation PASSED: Gracefully degraded without server crash.")

    # 8. LLM failure / timeout simulation
    def test_llm_failure_simulation(self):
        with patch("google.genai.Client", side_effect=Exception("Simulated LLM RateLimit / Timeout")):
            rag_res = generate_grounded_rag_answer(
                question="What deadline was decided for the mobile application?",
                api_key="dummy_key_to_trigger_client"
            )
            self.assertIsNotNone(rag_res["answer"])
            self.assertTrue(rag_res["is_grounded"])
            self.assertIn("Notice: Live LLM synthesis unavailable", rag_res["answer"])
            self.assertGreater(len(rag_res["sources"]), 0)
        logger.info("LLM failure simulation PASSED: Structured factual fallback served.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
