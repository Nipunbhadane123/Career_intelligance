import unittest
import time
import logging
from fastapi.testclient import TestClient
from api import app
from database import SessionLocal, Meeting, User

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityTest")

class TestUserAccessAndSecurity(unittest.TestCase):
    """
    Task 7: User Access & Security Validation
    Validates authentication, session management, user-specific meetings,
    and ensures strict cross-tenant isolation (User A cannot access User B's private meetings).
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        timestamp = int(time.time() * 1000)
        cls.user_a_name = f"user_alpha_{timestamp}"
        cls.user_b_name = f"user_beta_{timestamp}"
        cls.password = "StrongPass2026!"

        # Register User A
        r_a = cls.client.post("/auth/register", json={"username": cls.user_a_name, "password": cls.password})
        assert r_a.status_code == 201, f"User A registration failed: {r_a.text}"

        # Login User A
        r_login_a = cls.client.post("/auth/login", json={"username": cls.user_a_name, "password": cls.password})
        assert r_login_a.status_code == 200
        cls.token_a = r_login_a.json()["access_token"]
        cls.user_a_id = r_login_a.json()["user_id"]

        # Register User B
        r_b = cls.client.post("/auth/register", json={"username": cls.user_b_name, "password": cls.password})
        assert r_b.status_code == 201, f"User B registration failed: {r_b.text}"

        # Login User B
        r_login_b = cls.client.post("/auth/login", json={"username": cls.user_b_name, "password": cls.password})
        assert r_login_b.status_code == 200
        cls.token_b = r_login_b.json()["access_token"]
        cls.user_b_id = r_login_b.json()["user_id"]

        # Create a private meeting owned by User A
        db = SessionLocal()
        try:
            m = Meeting(
                filename="confidential_user_a_meeting.mp4",
                title="Confidential User A Strategy",
                transcript="Sarah: Secret internal budget discussion.",
                summary="Confidential strategy meeting with sensitive budget data.",
                user_id=cls.user_a_id,
                platform="upload"
            )
            db.add(m)
            db.commit()
            db.refresh(m)
            cls.private_meeting_id = m.id
        finally:
            db.close()

    def test_01_user_registration_validation(self):
        """Test registration constraints: short username, short password, duplicate username."""
        # Short username
        r = self.client.post("/auth/register", json={"username": "ab", "password": "validPassword1"})
        self.assertIn(r.status_code, [400, 422])

        # Short password
        r = self.client.post("/auth/register", json={"username": "valid_user", "password": "123"})
        self.assertIn(r.status_code, [400, 422])

        # Duplicate username
        r = self.client.post("/auth/register", json={"username": self.user_a_name, "password": self.password})
        self.assertEqual(r.status_code, 400)
        self.assertIn("already exists", r.json()["error"])
        logger.info("Registration security validation PASSED.")

    def test_02_login_and_credential_verification(self):
        """Test login with valid vs invalid passwords."""
        # Valid login
        r = self.client.post("/auth/login", json={"username": self.user_a_name, "password": self.password})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["username"], self.user_a_name)

        # Invalid password
        r_bad = self.client.post("/auth/login", json={"username": self.user_a_name, "password": "WrongPassword"})
        self.assertEqual(r_bad.status_code, 401)
        self.assertIn("Invalid username or password", r_bad.json()["error"])
        logger.info("Login & credential verification PASSED.")

    def test_03_session_management_and_auth_me(self):
        """Test Bearer token verification at /auth/me."""
        # Authenticated with Token A
        r = self.client.get("/auth/me", headers={"Authorization": f"Bearer {self.token_a}"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["username"], self.user_a_name)

        # Bad / expired token
        r_bad = self.client.get("/auth/me", headers={"Authorization": "Bearer invalid_token_12345"})
        self.assertEqual(r_bad.status_code, 401)
        logger.info("Session management & /auth/me verification PASSED.")

    def test_04_owner_can_access_own_meeting(self):
        """User A should successfully access their own private meeting."""
        headers = {"Authorization": f"Bearer {self.token_a}"}
        r = self.client.get(f"/meetings/{self.private_meeting_id}", headers=headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["id"], self.private_meeting_id)
        self.assertEqual(data["filename"], "confidential_user_a_meeting.mp4")
        logger.info("Owner meeting access PASSED.")

    def test_05_unauthorized_user_blocked_from_private_meeting(self):
        """User B MUST receive 403 Forbidden when trying to access User A's private meeting."""
        headers = {"Authorization": f"Bearer {self.token_b}"}
        r = self.client.get(f"/meetings/{self.private_meeting_id}", headers=headers)
        self.assertEqual(r.status_code, 403)
        self.assertIn("Access denied", r.json()["error"])
        logger.info("Cross-user private meeting isolation PASSED (403 Forbidden returned).")

    def test_06_unauthorized_report_export_blocked(self):
        """User B MUST receive 403 Forbidden when trying to export PDF/CSV of User A's meeting."""
        headers = {"Authorization": f"Bearer {self.token_b}"}
        
        # PDF export blocked
        r_pdf = self.client.get(f"/reports/pdf/{self.private_meeting_id}", headers=headers)
        self.assertEqual(r_pdf.status_code, 403)

        # CSV export blocked
        r_csv = self.client.get(f"/reports/csv/{self.private_meeting_id}", headers=headers)
        self.assertEqual(r_csv.status_code, 403)
        logger.info("Unauthorized report export blocking PASSED (403 Forbidden).")

    def test_07_unauthorized_meeting_deletion_blocked(self):
        """User B MUST receive 403 Forbidden when trying to delete User A's meeting."""
        headers = {"Authorization": f"Bearer {self.token_b}"}
        r = self.client.delete(f"/meetings/{self.private_meeting_id}", headers=headers)
        self.assertEqual(r.status_code, 403)
        logger.info("Unauthorized meeting deletion blocking PASSED (403 Forbidden).")

if __name__ == "__main__":
    unittest.main()
