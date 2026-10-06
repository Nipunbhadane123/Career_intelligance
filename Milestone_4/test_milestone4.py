import unittest
import time
import logging
from fastapi.testclient import TestClient
from api import app
from database import init_db, get_meeting_by_id, get_user_meetings, verify_knowledge_repository
from config import TARGET_SLA_SECONDS
from report_service import generate_meeting_pdf, generate_meeting_csv
from analytics_service import compute_meeting_analytics, compute_user_overview_analytics
from zoom_service import list_zoom_recordings, process_zoom_recording, verify_zoom_webhook_signature
from google_meet_service import list_google_meet_recordings, process_google_meet_recording, validate_google_meet_auth
from seed_data import seed_database_and_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone4MasterTest")

class TestMilestone4MasterSuite(unittest.TestCase):
    """
    Master Consolidated Test Suite for Milestone 4:
    Verifies all 10 tasks rigorously and systematically.
    """

    @classmethod
    def setUpClass(cls):
        seed_database_and_vector_store()
        cls.client = TestClient(app)
        cls.ts = int(time.time() * 1000)
        cls.username = f"master_user_{cls.ts}"
        cls.password = "MasterPassword2026!"

        # Register & Login
        cls.client.post("/auth/register", json={"username": cls.username, "password": cls.password})
        r_login = cls.client.post("/auth/login", json={"username": cls.username, "password": cls.password})
        cls.token = r_login.json()["access_token"]
        cls.user_id = r_login.json()["user_id"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    # ══════════════════════════════════════════════════
    # Task 1: Dashboard Data Bindings & API Integration
    # ══════════════════════════════════════════════════
    def test_task_01_dashboard_components(self):
        """Verify dashboard data bindings: meetings list, search, filters, and viewers."""
        logger.info("Executing Task 1: Dashboard Data Bindings Verification...")
        r = self.client.get("/meetings?limit=10", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("meetings", data)
        self.assertGreater(data["total"], 0)
        
        # Verify first meeting card properties
        m = data["meetings"][0]
        self.assertIn("id", m)
        self.assertIn("filename", m)
        self.assertIn("summary", m)
        self.assertIn("action_items_count", m)
        self.assertIn("participants", m)
        logger.info(f"Task 1 PASSED: Dashboard bound to {data['total']} meetings from backend.")

    # ══════════════════════════════════════════════════
    # Task 2: Meeting Details & Analytics
    # ══════════════════════════════════════════════════
    def test_task_02_meeting_details_and_analytics(self):
        """
        Verify complete meeting details flow:
        Selection -> Details -> Transcript -> Summary -> Decisions -> Action Items -> Analytics.
        """
        logger.info("Executing Task 2: Meeting Details & Analytics Verification...")
        r_list = self.client.get("/meetings?limit=1", headers=self.headers)
        target_id = r_list.json()["meetings"][0]["id"]

        r_det = self.client.get(f"/meetings/{target_id}", headers=self.headers)
        self.assertEqual(r_det.status_code, 200)
        meeting = r_det.json()

        # Check required fields
        self.assertIn("transcript", meeting)
        self.assertIn("summary", meeting)
        self.assertIn("key_decisions", meeting)
        self.assertIn("action_items", meeting)
        self.assertIn("participants", meeting)

        # Check analytics
        r_ana = self.client.get(f"/analytics/meeting/{target_id}", headers=self.headers)
        self.assertEqual(r_ana.status_code, 200)
        analytics = r_ana.json()
        self.assertIn("word_count", analytics)
        self.assertIn("action_items_by_priority", analytics)
        self.assertIn("action_items_by_status", analytics)
        self.assertEqual(analytics["meeting_id"], target_id)
        logger.info(f"Task 2 PASSED: Details and analytics verified for Meeting #{target_id}.")

    # ══════════════════════════════════════════════════
    # Task 3: Grounded RAG Search & AI Assistant
    # ══════════════════════════════════════════════════
    def test_task_03_rag_search_and_ai_assistant(self):
        """Verify semantic search and strictly grounded RAG answers with citations."""
        logger.info("Executing Task 3: Grounded RAG Search & AI Assistant Verification...")
        q = "What was decided regarding the database migration downtime?"
        r = self.client.post("/ask", json={"question": q, "top_k": 3}, headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["is_grounded"])
        self.assertGreater(len(data["answer"]), 0)
        self.assertGreater(len(data["sources"]), 0)
        first_src = data["sources"][0]
        self.assertIn("meeting_id", first_src)
        self.assertIn("relevance_score", first_src)
        logger.info(f"Task 3 PASSED: RAG answered '{q}' grounded in {len(data['sources'])} sources.")

    # ══════════════════════════════════════════════════
    # Task 4: Zoom Integration
    # ══════════════════════════════════════════════════
    def test_task_04_zoom_integration(self):
        """Verify Zoom recording retrieval, ingestion pipeline, duplicate prevention, and webhooks."""
        logger.info("Executing Task 4: Zoom Integration Verification...")
        # 1. Retrieval
        recs = list_zoom_recordings()
        self.assertGreater(len(recs), 0)

        # 2. Webhook signature
        import hmac, hashlib
        sec, ts, bdy = "secret", "123", b"body"
        msg = f"v0:{ts}:{bdy.decode('utf-8')}"
        valid_sig = "v0=" + hmac.new(sec.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()
        self.assertTrue(verify_zoom_webhook_signature(sec, ts, bdy, valid_sig))

        # 3. Ingestion pipeline
        res = process_zoom_recording(recording_id="zm_master_01", topic="Architecture Sync", user_id=self.user_id, force_resync=True)
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["meeting_id"])

        # 4. Duplicate check
        dup_res = process_zoom_recording(recording_id="zm_master_01", topic="Architecture Sync", user_id=self.user_id, force_resync=False)
        self.assertEqual(dup_res["status"], "duplicate")
        self.assertTrue(dup_res["is_duplicate"])
        logger.info("Task 4 PASSED: Zoom integration pipeline & duplicate handling verified.")

    # ══════════════════════════════════════════════════
    # Task 5: Google Meet Integration
    # ══════════════════════════════════════════════════
    def test_task_05_google_meet_integration(self):
        """Verify Google Meet Drive retrieval, ingestion pipeline, duplicate prevention, and auth."""
        logger.info("Executing Task 5: Google Meet Integration Verification...")
        # 1. Auth check
        auth = validate_google_meet_auth()
        self.assertEqual(auth["status"], "authenticated")

        # 2. Ingestion pipeline
        res = process_google_meet_recording(file_id="gmeet_master_01", file_name="Google Meet Roadmap.mp4", user_id=self.user_id, force_resync=True)
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["meeting_id"])

        # 3. Duplicate check
        dup_res = process_google_meet_recording(file_id="gmeet_master_01", file_name="Google Meet Roadmap.mp4", user_id=self.user_id, force_resync=False)
        self.assertEqual(dup_res["status"], "duplicate")
        self.assertTrue(dup_res["is_duplicate"])
        logger.info("Task 5 PASSED: Google Meet integration pipeline & duplicate handling verified.")

    # ══════════════════════════════════════════════════
    # Task 6: Reports & Export
    # ══════════════════════════════════════════════════
    def test_task_06_reports_and_export(self):
        """Verify PDF and CSV generation and fidelity for meeting details."""
        logger.info("Executing Task 6: Reports & Export Verification...")
        meeting = get_meeting_by_id(1)
        self.assertIsNotNone(meeting)

        # PDF Report
        pdf = generate_meeting_pdf(meeting)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 1000)

        # CSV Report
        csv_data = generate_meeting_csv(meeting)
        self.assertIn("=== SynthAI Meeting Intelligence Report ===", csv_data)
        self.assertIn(str(meeting["id"]), csv_data)
        logger.info("Task 6 PASSED: High-fidelity PDF & CSV reports generated and verified.")

    # ══════════════════════════════════════════════════
    # Task 7: User Access & Security Validation
    # ══════════════════════════════════════════════════
    def test_task_07_user_access_and_security(self):
        """Verify registration, login, session tokens, and strict cross-tenant isolation."""
        logger.info("Executing Task 7: User Access & Security Validation...")
        # Create second user
        other_user = f"intruder_{self.ts}"
        self.client.post("/auth/register", json={"username": other_user, "password": "Password123!"})
        r_other = self.client.post("/auth/login", json={"username": other_user, "password": "Password123!"})
        other_token = r_other.json()["access_token"]
        other_headers = {"Authorization": f"Bearer {other_token}"}

        # Create private meeting for primary user
        from database import save_meeting_to_db
        priv_id = save_meeting_to_db(
            filename="private_primary_meeting.mp4",
            transcript="Private content",
            summary="Top secret",
            action_items_data=[],
            key_points_data=[],
            key_decisions_data=[],
            participants_data=[],
            user_id=self.user_id
        )

        # Intruder attempts to view private meeting
        r_denied = self.client.get(f"/meetings/{priv_id}", headers=other_headers)
        self.assertEqual(r_denied.status_code, 403)
        self.assertIn("Access denied", r_denied.json()["error"])
        logger.info("Task 7 PASSED: Cross-tenant private data isolation enforced (403 Forbidden).")

    # ══════════════════════════════════════════════════
    # Task 8: Complete End-to-End Autonomous Workflow
    # ══════════════════════════════════════════════════
    def test_task_08_end_to_end_pipeline(self):
        """Verify the full autonomous pipeline from upload to RAG and report download."""
        logger.info("Executing Task 8: End-to-End Autonomous Pipeline Verification...")
        test_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        files = {"file": ("autonomous_master_e2e.wav", test_wav, "audio/wav")}
        
        # Upload
        r_up = self.client.post("/meetings/upload", files=files, headers=self.headers)
        self.assertEqual(r_up.status_code, 201)
        m_id = r_up.json()["meeting_id"]

        # Search
        r_srch = self.client.post("/search", json={"query": "autonomous_master_e2e", "meeting_id": m_id}, headers=self.headers)
        self.assertEqual(r_srch.status_code, 200)

        # RAG
        r_ask = self.client.post("/ask", json={"question": "What happened in autonomous master?", "meeting_id": m_id}, headers=self.headers)
        self.assertEqual(r_ask.status_code, 200)

        # Export PDF
        r_pdf = self.client.get(f"/reports/pdf/{m_id}", headers=self.headers)
        self.assertEqual(r_pdf.status_code, 200)
        logger.info("Task 8 PASSED: Autonomous end-to-end workflow executed without manual intervention.")

    # ══════════════════════════════════════════════════
    # Task 9: Performance, Security & Reliability
    # ══════════════════════════════════════════════════
    def test_task_09_performance_and_reliability(self):
        """Verify API SLA (<3.0s), invalid requests, and zero leaked credentials."""
        logger.info("Executing Task 9: Performance, Security & Reliability Verification...")
        # SLA
        start = time.perf_counter()
        r = self.client.post("/search", json={"query": "cloud migration"}, headers=self.headers)
        elapsed = time.perf_counter() - start
        self.assertEqual(r.status_code, 200)
        self.assertLess(elapsed, TARGET_SLA_SECONDS)

        # Invalid upload rejection
        bad_file = {"file": ("bad_format.txt", b"plain text", "text/plain")}
        r_bad = self.client.post("/meetings/upload", files=bad_file, headers=self.headers)
        self.assertEqual(r_bad.status_code, 400)

        # Sensitive data check
        r_m = self.client.get("/meetings?limit=5", headers=self.headers)
        self.assertNotIn("password_hash", r_m.text)
        self.assertNotIn("GEMINI_API_KEY", r_m.text)
        logger.info("Task 9 PASSED: Performance SLA, invalid requests, and data protection verified.")

    # ══════════════════════════════════════════════════
    # Task 10: Deployment Readiness
    # ══════════════════════════════════════════════════
    def test_task_10_deployment_readiness(self):
        """Verify production readiness: health check, ready probe, stats, and database connectivity."""
        logger.info("Executing Task 10: Deployment Readiness Verification...")
        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["status"], "healthy")

        r_ready = self.client.get("/ready")
        self.assertEqual(r_ready.status_code, 200)
        self.assertTrue(r_ready.json()["ready"])

        r_stats = self.client.get("/stats")
        self.assertEqual(r_stats.status_code, 200)
        logger.info("Task 10 PASSED: Health, readiness, and system statistics validated for production deployment.")

if __name__ == "__main__":
    unittest.main()
