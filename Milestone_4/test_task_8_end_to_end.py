import unittest
import time
import logging
from fastapi.testclient import TestClient
from api import app
from database import get_meeting_by_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EndToEndTest")

class TestEndToEndWorkflow(unittest.TestCase):
    """
    Task 8: Complete End-to-End Testing
    Tests the complete autonomous application pipeline from:
    User Login -> Upload/Import Meeting -> Audio Validation -> Whisper Transcription ->
    Transcript Storage -> LLM Summary -> Action Item Extraction -> Participant Assignment ->
    Knowledge Repository -> Embeddings -> Vector Database -> RAG Search -> FastAPI ->
    PDF / CSV Report.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        timestamp = int(time.time() * 1000)
        cls.username = f"e2e_user_{timestamp}"
        cls.password = "AutonomousE2EPassword123!"

    def test_complete_autonomous_pipeline(self):
        logger.info("Starting Task 8: Complete End-to-End Workflow Validation...")

        # Step 1: User Registration
        r_reg = self.client.post("/auth/register", json={"username": self.username, "password": self.password})
        self.assertEqual(r_reg.status_code, 201)

        # Step 2: User Login
        r_login = self.client.post("/auth/login", json={"username": self.username, "password": self.password})
        self.assertEqual(r_login.status_code, 200)
        token = r_login.json()["access_token"]
        user_id = r_login.json()["user_id"]
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"Step 1 & 2 PASSED: User '{self.username}' authenticated with token.")

        # Step 3 & 4 & 5: Upload / Import Meeting with Audio Validation & Ingestion Pipeline
        # Simulating multipart audio upload: audio/wav file
        test_wav_content = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        files = {"file": ("autonomous_e2e_strategy.wav", test_wav_content, "audio/wav")}
        data = {"title": "Autonomous End-to-End Pipeline Review"}

        r_upload = self.client.post("/meetings/upload", files=files, data=data, headers=headers)
        self.assertEqual(r_upload.status_code, 201)
        upload_data = r_upload.json()
        meeting_id = upload_data["meeting_id"]
        self.assertIsNotNone(meeting_id)
        logger.info(f"Step 3-6 PASSED: Audio validated, transcribed, and saved as Meeting #{meeting_id}.")

        # Step 7-10: Verify Transcript Storage, LLM Summary, Action Items, Participant Assignment
        meeting = get_meeting_by_id(meeting_id)
        self.assertIsNotNone(meeting)
        self.assertIn("transcript", meeting)
        self.assertIn("summary", meeting)
        self.assertGreater(len(meeting["action_items"]), 0, "Action items must be extracted")
        self.assertGreater(len(meeting["participants"]), 0, "Participants must be assigned")
        logger.info(f"Step 7-10 PASSED: Relational graph verified with {len(meeting['action_items'])} action items and {len(meeting['participants'])} participants.")

        # Step 11-12: Embeddings & Vector Database Indexing
        # Verify vector store can find the new meeting via semantic search
        time.sleep(0.5)
        r_search = self.client.post("/search", json={
            "query": "Autonomous End-to-End Pipeline deliverable",
            "meeting_id": meeting_id,
            "top_k": 3
        }, headers=headers)
        self.assertEqual(r_search.status_code, 200)
        search_data = r_search.json()
        self.assertGreater(search_data["total_results"], 0)
        logger.info(f"Step 11-12 PASSED: Embeddings generated and vector search retrieved {search_data['total_results']} units.")

        # Step 13: RAG Search & AI Assistant
        r_ask = self.client.post("/ask", json={
            "question": "What actions were discussed in the Autonomous End-to-End meeting?",
            "meeting_id": meeting_id,
            "top_k": 3
        }, headers=headers)
        self.assertEqual(r_ask.status_code, 200)
        rag_data = r_ask.json()
        self.assertGreater(len(rag_data["answer"]), 0)
        self.assertTrue(rag_data["is_grounded"])
        self.assertGreater(len(rag_data["sources"]), 0)
        logger.info(f"Step 13 PASSED: Grounded RAG answer generated with {len(rag_data['sources'])} cited sources.")

        # Step 14: PDF & CSV Report Generation
        r_pdf = self.client.get(f"/reports/pdf/{meeting_id}", headers=headers)
        self.assertEqual(r_pdf.status_code, 200)
        self.assertTrue(r_pdf.content.startswith(b"%PDF"))

        r_csv = self.client.get(f"/reports/csv/{meeting_id}", headers=headers)
        self.assertEqual(r_csv.status_code, 200)
        self.assertIn("=== SynthAI Meeting Intelligence Report ===", r_csv.text)
        self.assertIn(str(meeting_id), r_csv.text)
        logger.info("Step 14 PASSED: PDF and CSV reports generated and verified.")

        logger.info("TASK 8 PASSED: Complete end-to-end workflow executed seamlessly without manual intervention.")

if __name__ == "__main__":
    unittest.main()
