import os
import sys
import logging
from database import SessionLocal, init_db, get_all_meetings, create_user
from models import Meeting, Participant, ActionItem, KeyPoint, KeyDecision, IntegrationSync, User
from embedding_service import generate_meeting_embeddings
from vector_store import insert_meeting_vectors, get_vector_store_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED_USERS = [
    {"username": "admin", "password": "adminpassword123", "role": "admin"},
    {"username": "alice", "password": "alicepassword123", "role": "member"},
    {"username": "bob", "password": "bobpassword123", "role": "member"},
]

SAMPLE_MEETINGS = [
    {
        "filename": "database_migration_sync.mp4",
        "title": "Database Migration Sync",
        "platform": "upload",
        "duration_seconds": 1800.0,
        "summary": "The engineering team reviewed and finalized the database migration plan from legacy MySQL to Amazon Aurora PostgreSQL. A 2-hour maintenance window was scheduled for Sunday at 2 AM. Automated rollback scripts and staging dry runs were assigned to ensure zero data loss.",
        "transcript": """[00:00] Alice: Welcome everyone. Today our primary topic is the database migration strategy.
[00:15] Bob: We evaluated migrating our production database from legacy on-premises MySQL to Amazon Aurora PostgreSQL.
[00:45] Alice: What is the planned downtime window for the database migration?
[01:00] Bob: We will schedule a 2-hour maintenance window on Sunday at 2 AM.
[01:20] Charlie: I will prepare the automated migration rollback script by Friday at 5 PM.
[01:45] Alice: Perfect. The decision is approved: we migrate to Aurora PostgreSQL during the next maintenance window.""",
        "key_decisions": [
            "Approved the migration of the production database from legacy MySQL to Amazon Aurora PostgreSQL during the Sunday 2 AM maintenance window.",
            "Adopted zero-data-loss replication with an automated rollback fallback procedure."
        ],
        "key_points": [
            "Database migration will require a 2-hour scheduled maintenance window.",
            "Aurora PostgreSQL provides automated replication and cross-region failover."
        ],
        "participants": ["Alice", "Bob", "Charlie"],
        "action_items": [
            {
                "description": "Prepare the automated database migration rollback script and test recovery procedures.",
                "assigned_participant": "Charlie",
                "deadline": "Friday 5 PM",
                "priority": "High",
                "status": "Pending"
            },
            {
                "description": "Run a staging dry-run database migration on the staging database cluster.",
                "assigned_participant": "Bob",
                "deadline": "Next Tuesday",
                "priority": "High",
                "status": "Completed"
            }
        ]
    },
    {
        "filename": "mobile_application_release_planning.mp4",
        "title": "Mobile Application Release Planning",
        "platform": "upload",
        "duration_seconds": 2400.0,
        "summary": "The product and engineering teams established release milestones for the new mobile application. The official deadline decided for the mobile application beta release is October 15, 2026, followed by general release on November 1, 2026. UI testing and TestFlight submission tasks were assigned.",
        "transcript": """[00:00] David: Thanks for joining the mobile application roadmap meeting.
[00:10] Elena: Let's discuss our release targets for the iOS and Android mobile application.
[00:30] David: What deadline was decided for the mobile application beta release?
[00:45] Elena: The deadline decided for the mobile application beta release is October 15, 2026, with the general public release scheduled for November 1, 2026.
[01:10] Frank: I will finalize the mobile application UI components and accessibility testing by October 10, 2026.
[01:35] Elena: Great, I will submit the mobile application build to Apple TestFlight by October 12, 2026.""",
        "key_decisions": [
            "The official deadline decided for the mobile application beta release is October 15, 2026, and the general public launch is November 1, 2026."
        ],
        "key_points": [
            "Beta testers will receive the mobile application build on October 15, 2026.",
            "Feature freeze for the mobile application is set for October 1, 2026."
        ],
        "participants": ["David", "Elena", "Frank"],
        "action_items": [
            {
                "description": "Finalize the mobile application UI components and accessibility testing.",
                "assigned_participant": "Frank",
                "deadline": "October 10, 2026",
                "priority": "High",
                "status": "In Progress"
            },
            {
                "description": "Submit the mobile application build to Apple TestFlight.",
                "assigned_participant": "Elena",
                "deadline": "October 12, 2026",
                "priority": "High",
                "status": "Pending"
            }
        ]
    },
    {
        "filename": "zoom_q3_cloud_review.mp4",
        "title": "Zoom: Q3 Cloud Infrastructure Review",
        "platform": "zoom",
        "external_id": "zm_rec_982341",
        "duration_seconds": 2700.0,
        "summary": "Engineering review of Kubernetes cluster migration and AWS Aurora database switchover. Zero data loss verified with 35% latency drop. Production deployment approved for October 5th.",
        "transcript": """[00:00] Sarah: Welcome everyone to the Zoom architecture review. Today we review our Kubernetes cluster migration.
[00:20] David: The database migration completed last weekend with zero data loss. Latency dropped by 35%.
[00:40] Elena: Does this unblock the mobile app release?
[00:55] David: Yes, the new API endpoints are now stabilized. I will finalize the production monitoring alerts by Friday.
[01:20] Sarah: Approved. We execute production deployment on October 5th.""",
        "key_decisions": [
            "Approved production deployment date for October 5th.",
            "Standardized on AWS Aurora for all microservices."
        ],
        "key_points": [
            "Latency reduced by 35% across all REST endpoints.",
            "Kubernetes clusters running across 3 availability zones."
        ],
        "participants": ["Sarah", "David", "Elena"],
        "action_items": [
            {
                "description": "Finalize production monitoring alerts for Aurora endpoints",
                "assigned_participant": "David",
                "deadline": "2026-09-18",
                "priority": "High",
                "status": "Completed"
            }
        ]
    },
    {
        "filename": "gmeet_roadmap_2026.mp4",
        "title": "Google Meet: Product Strategy & Roadmap 2026",
        "platform": "google_meet",
        "external_id": "gmeet_drive_819234",
        "duration_seconds": 3200.0,
        "summary": "Finalization of SynthAI 2026 product strategy. Priorities include Google Drive Meet recording ingestion, multi-tenant vector indexing, and unified executive PDF report exports.",
        "transcript": """[00:00] Jessica: Good morning team. We are here to finalize the 2026 enterprise roadmap for SynthAI.
[00:25] Liam: From an engineering perspective, our priority is supporting multi-tenant vector databases and Google Meet/Zoom recording streaming.
[00:50] Carlos: I have completed the dashboard wireframes. Users will see a unified meetings feed with instant filters and PDF exports.
[01:15] Jessica: Approved. Carlos deliver the finalized UI components by September 25. Liam finish the Drive sync worker by October 1st.""",
        "key_decisions": [
            "Approved the 2026 Enterprise roadmap focusing on seamless Google Meet and Zoom ingestion.",
            "Adopted ReportLab for PDF executive reports and RFC-4180 for CSV exports."
        ],
        "key_points": [
            "Automated Google Drive Meet recordings folder polling scheduled for Q4.",
            "Sub-3-second semantic search latency SLA validated."
        ],
        "participants": ["Jessica", "Liam", "Carlos"],
        "action_items": [
            {
                "description": "Deliver finalized Streamlit UI components and design specs",
                "assigned_participant": "Carlos",
                "deadline": "2026-09-25",
                "priority": "High",
                "status": "Completed"
            },
            {
                "description": "Complete Google Drive background sync worker and OAuth refresh",
                "assigned_participant": "Liam",
                "deadline": "2026-10-01",
                "priority": "High",
                "status": "In Progress"
            }
        ]
    }
]

def seed_database_and_vector_store(force_reindex: bool = False):
    """
    Seeds initial users, sample meetings across platforms, and generates vector store embeddings.
    """
    logger.info("Initializing relational database tables...")
    init_db()
    db = SessionLocal()

    try:
        # Seed users
        for u in SEED_USERS:
            create_user(u["username"], u["password"], u["role"])

        # Fetch alice user id for user-specific meetings demonstration
        alice = db.query(User).filter(User.username == "alice").first()
        alice_id = alice.id if alice else None

        # Seed sample meetings
        for idx, s_data in enumerate(SAMPLE_MEETINGS):
            existing = db.query(Meeting).filter(Meeting.filename == s_data["filename"]).first()
            if not existing:
                logger.info(f"Seeding meeting: {s_data['filename']}")
                
                # Assign one meeting specifically to Alice for user-isolation testing
                m_user_id = alice_id if idx == 1 else None

                meeting = Meeting(
                    filename=s_data["filename"],
                    title=s_data.get("title", s_data["filename"]),
                    transcript=s_data["transcript"],
                    summary=s_data["summary"],
                    platform=s_data.get("platform", "upload"),
                    external_id=s_data.get("external_id"),
                    duration_seconds=s_data.get("duration_seconds", 1800.0),
                    user_id=m_user_id
                )
                db.add(meeting)
                db.flush()

                # Record sync for Zoom / Google Meet
                if s_data.get("platform") in ["zoom", "google_meet"] and s_data.get("external_id"):
                    sync_rec = IntegrationSync(
                        platform=s_data["platform"],
                        external_id=s_data["external_id"],
                        file_name=s_data["filename"],
                        file_hash=f"seed_hash_{s_data['external_id']}",
                        status="completed",
                        meeting_id=meeting.id
                    )
                    db.add(sync_rec)

                # Participants
                part_map = {}
                for p_name in s_data["participants"]:
                    part = db.query(Participant).filter(Participant.name == p_name).first()
                    if not part:
                        part = Participant(name=p_name)
                        db.add(part)
                        db.flush()
                    part_map[p_name] = part
                    if part not in meeting.participants:
                        meeting.participants.append(part)

                # Decisions
                for dec in s_data["key_decisions"]:
                    db.add(KeyDecision(meeting_id=meeting.id, decision=dec))

                # Key Points
                for kp in s_data["key_points"]:
                    db.add(KeyPoint(meeting_id=meeting.id, point=kp))

                # Action Items
                for ai in s_data["action_items"]:
                    p_name = ai.get("assigned_participant")
                    p_id = part_map[p_name].id if p_name in part_map else None
                    db.add(ActionItem(
                        meeting_id=meeting.id,
                        participant_id=p_id,
                        description=ai["description"],
                        deadline=ai.get("deadline"),
                        priority=ai.get("priority", "Medium"),
                        status=ai.get("status", "Pending")
                    ))

                db.commit()
            else:
                logger.info(f"Meeting already exists: {s_data['filename']}")

    finally:
        db.close()

    # Sync embeddings with ChromaDB
    all_meetings = get_all_meetings()
    logger.info(f"Indexing {len(all_meetings)} meetings into persistent ChromaDB store...")
    total_vectors = 0
    for m in all_meetings:
        vector_items = generate_meeting_embeddings(m)
        count = insert_meeting_vectors(m["id"], vector_items)
        total_vectors += count

    v_stats = get_vector_store_stats()
    logger.info(f"Vector Database Sync Complete! Total vectors in store: {v_stats.get('total_vectors', total_vectors)}")

if __name__ == "__main__":
    seed_database_and_vector_store()
