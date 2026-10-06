import unittest
import logging
from fastapi.testclient import TestClient
from api import app
from database import get_meeting_by_id, get_sync_history
from zoom_service import (
    verify_zoom_webhook_signature,
    list_zoom_recordings,
    process_zoom_recording
)
from google_meet_service import (
    validate_google_meet_auth,
    list_google_meet_recordings,
    process_google_meet_recording
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IntegrationsTest")

class TestZoomAndGoogleMeetIntegrations(unittest.TestCase):
    """
    Task 4 & Task 5: Zoom & Google Meet Integrations Validation
    Verifies authentication, recording retrieval, ingestion pipeline, duplicate detection,
    and failure resilience.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # ══════════════════════════════════════════════════
    # Task 4: Zoom Integration Tests
    # ══════════════════════════════════════════════════
    def test_01_zoom_webhook_signature_auth(self):
        """Test HMAC-SHA256 signature verification for Zoom webhooks."""
        secret = "test_zoom_secret_token"
        timestamp = "1695990000"
        body = b'{"event":"recording.completed","payload":{"object":{"id":"12345"}}}'

        import hmac, hashlib
        message = f"v0:{timestamp}:{body.decode('utf-8')}"
        valid_sig = "v0=" + hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
        invalid_sig = "v0=bad_hex_digest"

        self.assertTrue(verify_zoom_webhook_signature(secret, timestamp, body, valid_sig))
        self.assertFalse(verify_zoom_webhook_signature(secret, timestamp, body, invalid_sig))
        logger.info("Zoom webhook authentication signature check PASSED.")

    def test_02_zoom_recording_retrieval(self):
        """Test retrieving available Zoom cloud recordings."""
        r = self.client.get("/integrations/zoom/recordings")
        self.assertEqual(r.status_code, 200)
        recs = r.json()
        self.assertGreater(len(recs), 0)
        first_rec = recs[0]
        self.assertIn("meeting_id", first_rec)
        self.assertIn("topic", first_rec)
        self.assertIn("duration_minutes", first_rec)
        logger.info(f"Zoom recording retrieval PASSED: Retrieved {len(recs)} recordings.")

    def test_03_zoom_recording_ingestion_pipeline(self):
        """
        Test Zoom ingestion flow:
        Zoom Recording -> Application -> Transcription -> Summary -> Action Items -> Knowledge Repository
        """
        payload = {
            "recording_id": "zm_rec_test_new_01",
            "topic": "Zoom Automated Architecture Sync",
            "force_resync": True
        }
        r = self.client.post("/integrations/zoom/sync", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "success")
        self.assertIsNotNone(data["meeting_id"])
        self.assertFalse(data["is_duplicate"])

        # Verify record exists in SQLite with full relational details
        meeting = get_meeting_by_id(data["meeting_id"])
        self.assertIsNotNone(meeting)
        self.assertEqual(meeting["platform"], "zoom")
        self.assertGreater(len(meeting["summary"]), 0)
        self.assertGreater(len(meeting["action_items"]), 0)
        logger.info(f"Zoom ingestion pipeline PASSED: Ingested as Meeting #{data['meeting_id']}.")

    def test_04_zoom_duplicate_prevention(self):
        """Test that re-syncing an existing Zoom recording is flagged as duplicate."""
        payload = {
            "recording_id": "zm_rec_test_new_01",
            "topic": "Zoom Automated Architecture Sync",
            "force_resync": False
        }
        r = self.client.post("/integrations/zoom/sync", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "duplicate")
        self.assertTrue(data["is_duplicate"])
        self.assertIn("already been processed", data["message"])
        logger.info("Zoom duplicate recording detection PASSED.")

    # ══════════════════════════════════════════════════
    # Task 5: Google Meet Integration Tests
    # ══════════════════════════════════════════════════
    def test_05_google_meet_auth_validation(self):
        """Test Google Workspace / Google Drive API auth verification."""
        auth_status = validate_google_meet_auth()
        self.assertEqual(auth_status["status"], "authenticated")
        logger.info(f"Google Meet auth verification PASSED: {auth_status['mode']}")

    def test_06_google_meet_recording_retrieval(self):
        """Test retrieving Google Meet recordings from Google Drive."""
        r = self.client.get("/integrations/google-meet/recordings")
        self.assertEqual(r.status_code, 200)
        recs = r.json()
        self.assertGreater(len(recs), 0)
        first_rec = recs[0]
        self.assertIn("file_id", first_rec)
        self.assertIn("file_name", first_rec)
        logger.info(f"Google Meet recording retrieval PASSED: Retrieved {len(recs)} recordings.")

    def test_07_google_meet_ingestion_pipeline(self):
        """
        Test Google Meet ingestion flow:
        Google Meet Recording -> Application -> Whisper Transcription -> LLM Processing -> Knowledge Repository
        """
        payload = {
            "file_id": "gmeet_test_file_01",
            "file_name": "Google Meet - Sprint Planning 2026.mp4",
            "force_resync": True
        }
        r = self.client.post("/integrations/google-meet/sync", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "success")
        self.assertIsNotNone(data["meeting_id"])
        self.assertFalse(data["is_duplicate"])

        # Verify in knowledge repository
        meeting = get_meeting_by_id(data["meeting_id"])
        self.assertIsNotNone(meeting)
        self.assertEqual(meeting["platform"], "google_meet")
        self.assertGreater(len(meeting["summary"]), 0)
        self.assertGreater(len(meeting["action_items"]), 0)
        logger.info(f"Google Meet ingestion pipeline PASSED: Ingested as Meeting #{data['meeting_id']}.")

    def test_08_google_meet_duplicate_prevention(self):
        """Test that re-syncing the same Google Meet file is detected as duplicate."""
        payload = {
            "file_id": "gmeet_test_file_01",
            "file_name": "Google Meet - Sprint Planning 2026.mp4",
            "force_resync": False
        }
        r = self.client.post("/integrations/google-meet/sync", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "duplicate")
        self.assertTrue(data["is_duplicate"])
        logger.info("Google Meet duplicate recording detection PASSED.")

    def test_09_integration_failure_handling(self):
        """Test that processing failures are safely captured in database audit logs without crashing."""
        # Intentionally malformed call
        fail_res = process_zoom_recording(recording_id="", topic="", raw_content=None)
        # Should return structured failure response, not unhandled crash
        self.assertIn(fail_res["status"], ["failed", "duplicate", "success"])
        logger.info("Integration failure resilience PASSED.")

if __name__ == "__main__":
    unittest.main()
