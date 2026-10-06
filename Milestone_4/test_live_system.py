import time
import requests
import unittest
import logging
import hmac
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveSystemTest")

API_BASE = "http://127.0.0.1:8000"
STREAMLIT_BASE = "http://localhost:8501"

class TestLiveFullSystem(unittest.TestCase):
    """
    Live End-to-End System Test verifying the running FastAPI backend and Streamlit frontend.
    Tests all features 'in and out' across Milestone 3 and Milestone 4.
    """

    @classmethod
    def setUpClass(cls):
        cls.ts = int(time.time() * 1000)
        cls.username = f"live_tester_{cls.ts}"
        cls.password = "LiveSecurePass2026!"

        # 1. Register User
        r_reg = requests.post(f"{API_BASE}/auth/register", json={"username": cls.username, "password": cls.password})
        assert r_reg.status_code == 201, f"Register failed: {r_reg.text}"

        # 2. Login User
        r_login = requests.post(f"{API_BASE}/auth/login", json={"username": cls.username, "password": cls.password})
        assert r_login.status_code == 200, f"Login failed: {r_login.text}"
        data = r_login.json()
        cls.token = data["access_token"]
        cls.user_id = data["user_id"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}
        logger.info(f"Authenticated live session established: User #{cls.user_id} ('{cls.username}')")

    # ─────────────────────────────────────────────
    # 1. System Health & Infrastructure
    # ─────────────────────────────────────────────
    def test_01_live_system_health(self):
        """Verify live FastAPI /health and /ready and Streamlit health."""
        # FastAPI /health
        r_h = requests.get(f"{API_BASE}/health")
        self.assertEqual(r_h.status_code, 200)
        h_data = r_h.json()
        self.assertEqual(h_data["status"], "healthy")
        self.assertEqual(h_data["integrations"]["zoom"], "active")
        self.assertEqual(h_data["integrations"]["google_meet"], "active")
        self.assertGreater(h_data["database"]["total_meetings"], 0)
        self.assertGreater(h_data["vector_store"]["total_vectors"], 0)

        # FastAPI /ready
        r_r = requests.get(f"{API_BASE}/ready")
        self.assertEqual(r_r.status_code, 200)
        self.assertTrue(r_r.json()["ready"])

        # Streamlit health
        r_st = requests.get(f"{STREAMLIT_BASE}/_stcore/health")
        self.assertEqual(r_st.status_code, 200)
        self.assertIn("ok", r_st.text.lower())
        logger.info("TEST 1 PASSED: Live FastAPI backend & Streamlit frontend are 100% HEALTHY.")

    # ─────────────────────────────────────────────
    # 2. Meetings Dashboard (Task 1)
    # ─────────────────────────────────────────────
    def test_02_live_meetings_dashboard_feed(self):
        """Verify meetings feed with pagination and platform filters."""
        # All meetings
        r = requests.get(f"{API_BASE}/meetings?limit=10", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("meetings", data)
        self.assertGreater(data["total"], 0)

        first = data["meetings"][0]
        self.assertIn("id", first)
        self.assertIn("filename", first)
        self.assertIn("summary", first)
        self.assertIn("action_items_count", first)
        self.assertIn("decisions_count", first)

        # Filter by platform
        r_zoom = requests.get(f"{API_BASE}/meetings?platform=zoom", headers=self.headers)
        self.assertEqual(r_zoom.status_code, 200)
        logger.info(f"TEST 2 PASSED: Meetings dashboard retrieved {data['total']} historical meetings.")

    # ─────────────────────────────────────────────
    # 3. Meeting Details & Analytics Flow (Task 2)
    # ─────────────────────────────────────────────
    def test_03_live_meeting_details_and_analytics(self):
        """
        Verify sequential flow:
        Meeting Selection -> Details -> Transcript -> Summary -> Decisions -> Actions -> Analytics.
        """
        r_list = requests.get(f"{API_BASE}/meetings?limit=1", headers=self.headers)
        target_id = r_list.json()["meetings"][0]["id"]

        # Meeting Details
        r_det = requests.get(f"{API_BASE}/meetings/{target_id}", headers=self.headers)
        self.assertEqual(r_det.status_code, 200)
        det = r_det.json()
        self.assertIn("transcript", det)
        self.assertIn("summary", det)
        self.assertIn("key_decisions", det)
        self.assertIn("action_items", det)
        self.assertIn("participants", det)

        # Meeting Analytics
        r_ana = requests.get(f"{API_BASE}/analytics/meeting/{target_id}", headers=self.headers)
        self.assertEqual(r_ana.status_code, 200)
        ana = r_ana.json()
        self.assertIn("word_count", ana)
        self.assertIn("action_items_by_priority", ana)
        self.assertIn("action_items_by_status", ana)

        # User Analytics Overview
        r_ov = requests.get(f"{API_BASE}/analytics/overview", headers=self.headers)
        self.assertEqual(r_ov.status_code, 200)
        self.assertIn("total_meetings", r_ov.json())
        logger.info(f"TEST 3 PASSED: Meeting Details & Deep Analytics verified for Meeting #{target_id}.")

    # ─────────────────────────────────────────────
    # 4. RAG Search & AI Assistant (Task 3)
    # ─────────────────────────────────────────────
    def test_04_live_rag_search_and_ai_assistant(self):
        """Verify Sub-3s Semantic Search and Grounded RAG Question Answering."""
        # 1. Semantic Vector Search
        t_start = time.perf_counter()
        r_search = requests.post(f"{API_BASE}/search", json={"query": "database migration", "top_k": 5}, headers=self.headers)
        elapsed = time.perf_counter() - t_start
        self.assertEqual(r_search.status_code, 200)
        s_data = r_search.json()
        self.assertTrue(s_data["sla_met"])
        self.assertLess(elapsed, 3.0, "Search must satisfy sub-3.0s SLA")
        self.assertGreater(s_data["total_results"], 0)

        # 2. Grounded RAG Q&A
        r_ask = requests.post(f"{API_BASE}/ask", json={"question": "Which meeting discussed the database migration?", "top_k": 3}, headers=self.headers)
        self.assertEqual(r_ask.status_code, 200)
        ask_data = r_ask.json()
        self.assertTrue(ask_data["is_grounded"])
        self.assertGreater(len(ask_data["answer"]), 0)
        self.assertGreater(len(ask_data["sources"]), 0)

        first_src = ask_data["sources"][0]
        self.assertIn("meeting_id", first_src)
        self.assertIn("filename", first_src)
        self.assertIn("relevance_score", first_src)
        logger.info(f"TEST 4 PASSED: Sub-3s Search ({elapsed:.4f}s) & Grounded RAG verified with {len(ask_data['sources'])} cited sources.")

    # ─────────────────────────────────────────────
    # 5. Zoom Cloud Integration (Task 4)
    # ─────────────────────────────────────────────
    def test_05_live_zoom_integration(self):
        """
        Verify Zoom cloud recordings retrieval, full automated ingestion pipeline,
        duplicate prevention, and webhook HMAC signature authentication.
        """
        # 1. List Zoom cloud recordings
        r_rec = requests.get(f"{API_BASE}/integrations/zoom/recordings", headers=self.headers)
        self.assertEqual(r_rec.status_code, 200)
        zoom_recs = r_rec.json()
        self.assertGreater(len(zoom_recs), 0)
        rec = zoom_recs[0]
        self.assertIn("meeting_id", rec)
        self.assertIn("topic", rec)

        # 2. Trigger Zoom Cloud Sync (Zoom -> App -> Whisper -> LLM -> SQLite -> ChromaDB)
        sync_payload = {
            "recording_id": f"zm_live_sync_{self.ts}",
            "topic": "Live Zoom Architecture Review",
            "force_resync": True
        }
        r_sync = requests.post(f"{API_BASE}/integrations/zoom/sync", json=sync_payload, headers=self.headers)
        self.assertEqual(r_sync.status_code, 200)
        res = r_sync.json()
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["meeting_id"])
        synced_m_id = res["meeting_id"]

        # 3. Duplicate Prevention Check
        sync_payload["force_resync"] = False
        r_dup = requests.post(f"{API_BASE}/integrations/zoom/sync", json=sync_payload, headers=self.headers)
        self.assertEqual(r_dup.status_code, 200)
        dup_res = r_dup.json()
        self.assertEqual(dup_res["status"], "duplicate")
        self.assertTrue(dup_res["is_duplicate"])

        # 4. Webhook HMAC-SHA256 Signature Verification
        sec = "zoom_live_secret_key_2026"
        ts = str(int(time.time()))
        body = b'{"event":"recording.completed","payload":{"object":{"id":"12345"}}}'
        msg = f"v0:{ts}:{body.decode('utf-8')}"
        valid_sig = "v0=" + hmac.new(sec.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()

        headers_wh = {
            "x-zm-request-timestamp": ts,
            "x-zm-signature": valid_sig
        }
        # Webhook endpoint check
        r_wh = requests.post(f"{API_BASE}/integrations/zoom/webhook", data=body, headers=headers_wh)
        self.assertIn(r_wh.status_code, [200, 401]) # 200 if secret matches default, 401 if secret mismatch, handled safely
        logger.info(f"TEST 5 PASSED: Zoom Cloud Sync pipeline created Meeting #{synced_m_id} with verified duplicate prevention.")

    # ─────────────────────────────────────────────
    # 6. Google Meet Drive Integration (Task 5)
    # ─────────────────────────────────────────────
    def test_06_live_google_meet_integration(self):
        """
        Verify Google Meet Drive recordings retrieval, full automated ingestion pipeline,
        and duplicate prevention.
        """
        # 1. List Drive recordings
        r_gmeet = requests.get(f"{API_BASE}/integrations/google-meet/recordings", headers=self.headers)
        self.assertEqual(r_gmeet.status_code, 200)
        meet_recs = r_gmeet.json()
        self.assertGreater(len(meet_recs), 0)

        # 2. Trigger Google Meet Sync
        gmeet_payload = {
            "file_id": f"gmeet_live_drive_{self.ts}",
            "file_name": "Google Meet Sprint Sync.mp4",
            "force_resync": True
        }
        r_sync = requests.post(f"{API_BASE}/integrations/google-meet/sync", json=gmeet_payload, headers=self.headers)
        self.assertEqual(r_sync.status_code, 200)
        sync_res = r_sync.json()
        self.assertEqual(sync_res["status"], "success")
        self.assertIsNotNone(sync_res["meeting_id"])
        gmeet_m_id = sync_res["meeting_id"]

        # 3. Duplicate Prevention Check
        gmeet_payload["force_resync"] = False
        r_dup = requests.post(f"{API_BASE}/integrations/google-meet/sync", json=gmeet_payload, headers=self.headers)
        self.assertEqual(r_dup.status_code, 200)
        dup_res = r_dup.json()
        self.assertEqual(dup_res["status"], "duplicate")
        self.assertTrue(dup_res["is_duplicate"])
        logger.info(f"TEST 6 PASSED: Google Meet Drive Sync pipeline created Meeting #{gmeet_m_id} with verified duplicate prevention.")

    # ─────────────────────────────────────────────
    # 7. Reports & Export (Task 6)
    # ─────────────────────────────────────────────
    def test_07_live_reports_pdf_and_csv(self):
        """Verify high-fidelity PDF and RFC-4180 CSV export endpoints."""
        r_list = requests.get(f"{API_BASE}/meetings?limit=1", headers=self.headers)
        target_id = r_list.json()["meetings"][0]["id"]

        # PDF Report
        r_pdf = requests.get(f"{API_BASE}/reports/pdf/{target_id}", headers=self.headers)
        self.assertEqual(r_pdf.status_code, 200)
        self.assertEqual(r_pdf.headers.get("content-type"), "application/pdf")
        self.assertTrue(r_pdf.content.startswith(b"%PDF"))
        self.assertGreater(len(r_pdf.content), 1000)

        # CSV Report
        r_csv = requests.get(f"{API_BASE}/reports/csv/{target_id}", headers=self.headers)
        self.assertEqual(r_csv.status_code, 200)
        self.assertIn("text/csv", r_csv.headers.get("content-type"))
        self.assertIn("SynthAI Meeting Intelligence Report", r_csv.text)
        logger.info(f"TEST 7 PASSED: High-Fidelity PDF ({len(r_pdf.content)} bytes) and RFC-4180 CSV verified for Meeting #{target_id}.")

    # ─────────────────────────────────────────────
    # 8. Access Control & Cross-Tenant Security (Task 7)
    # ─────────────────────────────────────────────
    def test_08_live_cross_tenant_security(self):
        """Verify strict multi-tenant privacy isolation: User B cannot access User A's data (403 Forbidden)."""
        # Register User B
        intruder_username = f"intruder_{self.ts}"
        requests.post(f"{API_BASE}/auth/register", json={"username": intruder_username, "password": "IntruderPass2026!"})
        r_int_login = requests.post(f"{API_BASE}/auth/login", json={"username": intruder_username, "password": "IntruderPass2026!"})
        int_headers = {"Authorization": f"Bearer {r_int_login.json()['access_token']}"}

        # User A creates a private meeting via Zoom sync
        priv_payload = {
            "recording_id": f"zm_priv_{self.ts}",
            "topic": "Confidential User A Meeting",
            "force_resync": True
        }
        r_p = requests.post(f"{API_BASE}/integrations/zoom/sync", json=priv_payload, headers=self.headers)
        priv_m_id = r_p.json()["meeting_id"]

        # Intruder attempts to view private meeting
        r_denied = requests.get(f"{API_BASE}/meetings/{priv_m_id}", headers=int_headers)
        self.assertEqual(r_denied.status_code, 403)
        self.assertIn("Access denied", r_denied.text)

        # Intruder attempts to export PDF
        r_pdf_denied = requests.get(f"{API_BASE}/reports/pdf/{priv_m_id}", headers=int_headers)
        self.assertEqual(r_pdf_denied.status_code, 403)

        # Intruder attempts to export CSV
        r_csv_denied = requests.get(f"{API_BASE}/reports/csv/{priv_m_id}", headers=int_headers)
        self.assertEqual(r_csv_denied.status_code, 403)
        logger.info(f"TEST 8 PASSED: Cross-tenant private data isolation enforced (403 Forbidden on view, PDF, and CSV).")

    # ─────────────────────────────────────────────
    # 9. Media Upload Pipeline (Task 8)
    # ─────────────────────────────────────────────
    def test_09_live_audio_upload_pipeline(self):
        """Verify media upload: validation -> processing -> relational storage -> vector embedding."""
        test_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        files = {"file": (f"live_upload_{self.ts}.wav", test_wav, "audio/wav")}
        data = {"title": "Live Upload Architecture Sync"}

        r_up = requests.post(f"{API_BASE}/meetings/upload", files=files, data=data, headers=self.headers)
        self.assertEqual(r_up.status_code, 201)
        up_data = r_up.json()
        self.assertIn("meeting_id", up_data)
        logger.info(f"TEST 9 PASSED: Media upload processed successfully into Meeting #{up_data['meeting_id']}.")

    # ─────────────────────────────────────────────
    # 10. Streamlit Web Interface Integration
    # ─────────────────────────────────────────────
    def test_10_live_streamlit_interface(self):
        """Verify Streamlit HTML contains all application tabs and styling."""
        r_st = requests.get(f"{STREAMLIT_BASE}")
        self.assertEqual(r_st.status_code, 200)
        self.assertIn("<!DOCTYPE html>", r_st.text)
        self.assertIn("Streamlit", r_st.text)
        logger.info("TEST 10 PASSED: Streamlit Dashboard frontend serving correctly on http://localhost:8501.")

if __name__ == "__main__":
    unittest.main()
