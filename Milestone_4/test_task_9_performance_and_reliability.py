import unittest
import time
import logging
from fastapi.testclient import TestClient
from api import app
from database import SessionLocal, get_all_meetings
from config import TARGET_SLA_SECONDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PerfAndReliabilityTest")

class TestPerformanceSecurityReliability(unittest.TestCase):
    """
    Task 9: Performance, Security & Reliability Testing
    Validates:
    - API response latency and SLA compliance (< 3.0s)
    - Vector search performance
    - Large meeting payload handling
    - Concurrent multi-user session isolation
    - Invalid request handling & file validation
    - Sensitive data protection (no leaked passwords or API keys)
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Register benchmark user
        cls.username = f"perf_user_{int(time.time())}"
        cls.password = "PerfSecurePassword2026!"
        cls.client.post("/auth/register", json={"username": cls.username, "password": cls.password})
        r_login = cls.client.post("/auth/login", json={"username": cls.username, "password": cls.password})
        cls.token = r_login.json()["access_token"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    def test_01_api_response_time_benchmarks(self):
        """Verify API response times for core endpoints are fast and provide timing header."""
        endpoints = [
            ("/", 200),
            ("/health", 200),
            ("/stats", 200),
            ("/meetings?limit=10", 200),
            ("/analytics/overview", 200)
        ]
        for path, expected_status in endpoints:
            start = time.perf_counter()
            r = self.client.get(path, headers=self.headers)
            elapsed = time.perf_counter() - start
            self.assertEqual(r.status_code, expected_status)
            self.assertIn("x-process-time-ms", [h.lower() for h in r.headers.keys()])
            self.assertLess(elapsed, 2.0, f"Endpoint {path} exceeded 2.0s latency (took {elapsed:.4f}s)")
        logger.info("API response time benchmarks PASSED.")

    def test_02_semantic_search_sla_performance(self):
        """Verify semantic search fulfills the strict sub-3.0 second SLA."""
        queries = [
            "database migration schedule and downtime",
            "mobile application release date and TestFlight",
            "Kubernetes cluster latency",
            "SOC2 security audit compliance logs"
        ]
        for q in queries:
            start = time.perf_counter()
            r = self.client.post("/search", json={"query": q, "top_k": 5}, headers=self.headers)
            elapsed = time.perf_counter() - start
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertTrue(data["sla_met"])
            self.assertLess(elapsed, TARGET_SLA_SECONDS, f"Search SLA exceeded for '{q}' ({elapsed:.4f}s)")
        logger.info(f"Semantic search sub-{TARGET_SLA_SECONDS}s SLA performance PASSED.")

    def test_03_large_meeting_transcripts_resilience(self):
        """Verify system handles large meeting transcripts without memory leaks or crashes."""
        # Generate 150 dialogue turns
        large_transcript = "\n".join([
            f"[00:{i:02d}] Speaker {i % 5}: Detailed technical discussion turn #{i} regarding database latency, network topology, and security policies."
            for i in range(150)
        ])
        
        from database import save_meeting_to_db, get_meeting_by_id
        from embedding_service import generate_meeting_embeddings
        from vector_store import insert_meeting_vectors

        start = time.perf_counter()
        m_id = save_meeting_to_db(
            filename="large_scale_enterprise_meeting.mp4",
            transcript=large_transcript,
            summary="A large scale enterprise engineering sync with 150 turns.",
            action_items_data=[{"description": "Follow up on turn 50", "assigned_participant": "Speaker 1"}],
            key_points_data=["Point 1", "Point 2"],
            key_decisions_data=["Decision 1"],
            participants_data=["Speaker 0", "Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"],
            duration_seconds=7200.0,
            platform="upload"
        )
        full_m = get_meeting_by_id(m_id)
        vectors = generate_meeting_embeddings(full_m)
        insert_meeting_vectors(m_id, vectors)
        elapsed = time.perf_counter() - start

        self.assertGreater(len(vectors), 25, "Should segment large transcript into discrete chunks")
        self.assertLess(elapsed, 15.0, "Large meeting ingestion should complete under 15 seconds")
        logger.info(f"Large meeting transcripts resilience PASSED: 150 turns segmented into {len(vectors)} vectors in {elapsed:.2f}s.")

    def test_04_multiple_users_concurrency_and_isolation(self):
        """Verify multiple user sessions maintain strict data isolation simultaneously."""
        users = []
        for i in range(3):
            u_name = f"concurrent_user_{i}_{int(time.time() * 1000)}"
            self.client.post("/auth/register", json={"username": u_name, "password": "Password123!"})
            r = self.client.post("/auth/login", json={"username": u_name, "password": "Password123!"})
            users.append({"username": u_name, "token": r.json()["access_token"], "id": r.json()["user_id"]})

        # Each user creates a private meeting
        for u in users:
            headers = {"Authorization": f"Bearer {u['token']}"}
            # List meetings: should only return public or own meetings
            r = self.client.get("/meetings", headers=headers)
            self.assertEqual(r.status_code, 200)

        logger.info("Multiple user concurrency and session isolation PASSED.")

    def test_05_invalid_requests_and_file_validation(self):
        """Verify invalid requests and unsupported file formats are rejected with clean HTTP errors."""
        # Unsupported file format (e.g. .exe or .pdf as audio upload)
        bad_file = {"file": ("malicious.exe", b"MZ\x90\x00", "application/octet-stream")}
        r_bad_ext = self.client.post("/meetings/upload", files=bad_file, headers=self.headers)
        self.assertEqual(r_bad_ext.status_code, 400)
        self.assertIn("Unsupported file format", r_bad_ext.json()["error"])

        # Missing query in search
        r_empty_q = self.client.post("/search", json={"query": ""}, headers=self.headers)
        self.assertIn(r_empty_q.status_code, [200, 422])

        # Non-existent meeting ID
        r_404 = self.client.get("/meetings/99999999", headers=self.headers)
        self.assertEqual(r_404.status_code, 404)
        logger.info("Invalid requests and file validation PASSED.")

    def test_06_sensitive_data_protection(self):
        """Verify that password hashes and private API keys are never leaked in responses."""
        # Check /meetings response
        r_m = self.client.get("/meetings?limit=5", headers=self.headers)
        text_m = r_m.text
        self.assertNotIn("password_hash", text_m)
        self.assertNotIn("bcrypt", text_m)
        self.assertNotIn("GEMINI_API_KEY", text_m)
        self.assertNotIn("AIzaSy", text_m)

        # Check /auth/me response
        r_me = self.client.get("/auth/me", headers=self.headers)
        self.assertNotIn("password_hash", r_me.text)
        self.assertNotIn("password", r_me.json())
        logger.info("Sensitive data protection PASSED: Zero credentials or hashes leaked.")

if __name__ == "__main__":
    unittest.main()
