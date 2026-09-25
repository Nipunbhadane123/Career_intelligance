import unittest
import time
import logging
from fastapi.testclient import TestClient
from api import app, create_access_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("APIIntegrationTest")

class TestAPIIntegration(unittest.TestCase):
    """
    Task 6 & Task 9: End-to-End API Integration Testing
    Verifies that all API endpoints work seamlessly with the database, authentication,
    semantic search, and grounded RAG services.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_username = f"testuser_{int(time.time())}"
        cls.test_password = "SecretPassword123!"

    # 1. Root & Health Checks
    def test_01_root_and_health(self):
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertEqual(r_root.json()["status"], "online")

        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        health_data = r_health.json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertGreater(health_data["database"]["total_meetings"], 0)
        self.assertGreater(health_data["vector_store"]["total_vectors"], 0)
        logger.info(f"Health check PASSED: {health_data['database']['total_meetings']} meetings, {health_data['vector_store']['total_vectors']} vectors.")

    # 2. Stats
    def test_02_system_stats(self):
        r = self.client.get("/stats")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("vector_store", data)
        self.assertIn("knowledge_repository", data)
        self.assertGreater(data["knowledge_repository"]["total_meetings"], 0)
        logger.info("System stats PASSED.")

    # 3. Authentication: Register and Login
    def test_03_authentication_flow(self):
        # Register new user
        reg_payload = {"username": self.test_username, "password": self.test_password}
        r_reg = self.client.post("/auth/register", json=reg_payload)
        self.assertEqual(r_reg.status_code, 201)
        self.assertEqual(r_reg.json()["username"], self.test_username)

        # Duplicate register should fail with 400
        r_dup = self.client.post("/auth/register", json=reg_payload)
        self.assertEqual(r_dup.status_code, 400)
        self.assertIn("already exists", r_dup.json()["error"])

        # Login with correct credentials
        r_login = self.client.post("/auth/login", json=reg_payload)
        self.assertEqual(r_login.status_code, 200)
        auth_data = r_login.json()
        self.assertIn("access_token", auth_data)
        self.assertEqual(auth_data["username"], self.test_username)
        self.__class__.token = auth_data["access_token"]

        # Login with bad password should fail with 401
        r_bad = self.client.post("/auth/login", json={"username": self.test_username, "password": "WrongPassword"})
        self.assertEqual(r_bad.status_code, 401)
        logger.info("Authentication flow PASSED: Register, duplicate check, login, and invalid password verified.")

    # 4. GET /meetings
    def test_04_get_meetings(self):
        r = self.client.get("/meetings?limit=5")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("meetings", data)
        self.assertIn("total", data)
        self.assertGreater(data["total"], 0)
        self.assertLessEqual(len(data["meetings"]), 5)

        first_m = data["meetings"][0]
        self.assertIn("id", first_m)
        self.assertIn("filename", first_m)
        self.assertIn("summary", first_m)
        self.assertIn("action_items_count", first_m)
        logger.info(f"GET /meetings PASSED: Retrieved {len(data['meetings'])} of {data['total']} meetings.")

    # 5. GET /meetings/{id}
    def test_05_get_meeting_by_id(self):
        # Fetch first meeting id
        r_all = self.client.get("/meetings")
        first_id = r_all.json()["meetings"][0]["id"]

        r_detail = self.client.get(f"/meetings/{first_id}")
        self.assertEqual(r_detail.status_code, 200)
        detail = r_detail.json()
        self.assertEqual(detail["id"], first_id)
        self.assertIn("action_items", detail)
        self.assertIn("key_decisions", detail)
        self.assertIn("key_points", detail)
        self.assertIn("participants", detail)
        self.assertIn("transcript", detail)

        # 404 for non-existent meeting
        r_404 = self.client.get("/meetings/9999999")
        self.assertEqual(r_404.status_code, 404)
        self.assertIn("not found", r_404.json()["error"].lower())
        logger.info("GET /meetings/{id} and 404 handling PASSED.")

    # 6. POST /search (Semantic Search Engine)
    def test_06_post_search(self):
        payload = {
            "query": "Which meeting discussed the database migration?",
            "top_k": 5
        }
        start = time.perf_counter()
        r = self.client.post("/search", json=payload)
        elapsed = time.perf_counter() - start

        self.assertEqual(r.status_code, 200)
        res = r.json()
        self.assertTrue(res["sla_met"], f"SLA violated: {res.get('retrieval_time_seconds')}s")
        self.assertLess(elapsed, 3.0, "API HTTP round-trip must meet < 3s SLA")
        self.assertGreater(res["total_hits"], 0)
        
        # Verify database migration meeting ranked first
        top_m = res["matched_meetings"][0]["filename"]
        self.assertIn("migration", top_m.lower())
        logger.info(f"POST /search PASSED: Retrieved '{top_m}' in {res['retrieval_time_seconds']}s (HTTP: {elapsed:.2f}s).")

    # 7. GET /search
    def test_07_get_search(self):
        r = self.client.get("/search?query=mobile+application+release+deadline&top_k=4")
        self.assertEqual(r.status_code, 200)
        res = r.json()
        self.assertTrue(res["sla_met"])
        self.assertGreater(res["total_hits"], 0)
        top_m = res["matched_meetings"][0]["filename"]
        self.assertIn("mobile", top_m.lower())
        logger.info(f"GET /search PASSED: Top meeting '{top_m}'.")

    # 8. POST /ask (Grounded RAG Q&A)
    def test_08_post_ask_grounded_rag(self):
        payload = {
            "question": "What deadline was decided for the mobile application beta release?",
            "top_k": 5
        }
        r = self.client.post("/ask", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["question"], payload["question"])
        self.assertTrue(data["is_grounded"])
        self.assertGreater(len(data["sources"]), 0)
        self.assertLess(data["retrieval_time_seconds"], 3.0)

        # Check citations
        sources_text = " ".join([s.get("text", "") for s in data["sources"]]).lower()
        self.assertTrue("october 15" in sources_text or "october" in sources_text)
        logger.info(f"POST /ask PASSED: Grounded response generated with {len(data['sources'])} sources.")

    # 9. Validation & Error Handling
    def test_09_validation_error_handling(self):
        # Missing required parameter in GET /search
        r_err = self.client.get("/search")
        self.assertEqual(r_err.status_code, 422)
        err_data = r_err.json()
        self.assertEqual(err_data["status_code"], 422)
        self.assertIn("validation failed", err_data["error"].lower())
        logger.info("Error handling & validation schemas PASSED.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
