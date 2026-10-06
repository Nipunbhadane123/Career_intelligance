import unittest
import logging
from fastapi.testclient import TestClient
from api import app
from database import get_meeting_by_id
from report_service import generate_meeting_pdf, generate_meeting_csv, verify_report_contents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReportsTest")

class TestReportsAndExport(unittest.TestCase):
    """
    Task 6: Reports & Export Validation
    Tests PDF and CSV report generation for meeting details, summaries, decisions,
    action items, participants, and deadlines.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Fetch first available meeting
        r = cls.client.get("/meetings?limit=1")
        meetings = r.json()["meetings"]
        assert len(meetings) > 0, "At least one meeting must exist in database"
        cls.meeting_id = meetings[0]["id"]
        cls.meeting = get_meeting_by_id(cls.meeting_id)

    def test_01_pdf_report_generation(self):
        """Test PDF report generation: checks PDF magic bytes, headers, and file size."""
        pdf_bytes = generate_meeting_pdf(self.meeting)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000, "PDF should be non-trivial in size")
        # Check standard PDF file signature
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "File must start with %PDF magic header")
        logger.info(f"PDF report generation PASSED: Generated {len(pdf_bytes)} bytes.")

    def test_02_pdf_download_endpoint(self):
        """Test GET /reports/pdf/{meeting_id} API endpoint."""
        r = self.client.get(f"/reports/pdf/{self.meeting_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("content-type"), "application/pdf")
        self.assertIn("attachment", r.headers.get("content-disposition", ""))
        self.assertTrue(r.content.startswith(b"%PDF"))
        logger.info("PDF download endpoint PASSED.")

    def test_03_csv_report_generation(self):
        """Test CSV report generation: checks RFC-4180 CSV sections, headers, and values."""
        csv_text = generate_meeting_csv(self.meeting)
        self.assertIsInstance(csv_text, str)
        self.assertIn("=== SynthAI Meeting Intelligence Report ===", csv_text)
        self.assertIn("=== Executive Summary ===", csv_text)
        self.assertIn("=== Action Items ===", csv_text)
        self.assertIn("=== Key Decisions ===", csv_text)
        self.assertIn("=== Participants ===", csv_text)

        # Check meeting ID & filename fidelity
        self.assertIn(str(self.meeting_id), csv_text)
        self.assertIn(self.meeting["filename"], csv_text)
        logger.info("CSV report generation PASSED.")

    def test_04_csv_download_endpoint(self):
        """Test GET /reports/csv/{meeting_id} API endpoint."""
        r = self.client.get(f"/reports/csv/{self.meeting_id}")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/csv", r.headers.get("content-type", ""))
        self.assertIn("attachment", r.headers.get("content-disposition", ""))
        self.assertIn("=== SynthAI Meeting Intelligence Report ===", r.text)
        logger.info("CSV download endpoint PASSED.")

    def test_05_report_fidelity_verification(self):
        """Verify that exported report contains all required elements for the selected meeting."""
        csv_text = generate_meeting_csv(self.meeting)
        # Verify fidelity
        self.assertTrue(verify_report_contents(self.meeting, csv_text))
        
        # Verify action items and deadlines are in the CSV
        for ai in self.meeting.get("action_items", []):
            self.assertIn(ai["description"], csv_text)
            if ai.get("deadline"):
                self.assertIn(ai["deadline"], csv_text)

        # Verify key decisions are in the CSV
        for dec in self.meeting.get("key_decisions", []):
            self.assertIn(dec, csv_text)

        # Verify participants
        for p in self.meeting.get("participants", []):
            self.assertIn(p, csv_text)

        logger.info("Report data fidelity verification PASSED: All details, decisions, action items, participants, and deadlines verified.")

if __name__ == "__main__":
    unittest.main()
