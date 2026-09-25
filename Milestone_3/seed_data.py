import os
import sys
import logging
from database import SessionLocal, init_db, get_all_meetings, verify_knowledge_repository
from models import Meeting, Participant, ActionItem, KeyPoint, KeyDecision
from embedding_service import generate_meeting_embeddings
from vector_store import insert_meeting_vectors, get_vector_store_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_MEETINGS = [
    {
        "filename": "database_migration_sync.mp4",
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
                "status": "Pending"
            }
        ]
    },
    {
        "filename": "mobile_application_release_planning.mp4",
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
                "status": "Pending"
            },
            {
                "description": "Submit the mobile application build to Apple TestFlight.",
                "assigned_participant": "Elena",
                "deadline": "October 12, 2026",
                "priority": "High",
                "status": "Pending"
            }
        ]
    }
]

def seed_database_and_vector_store(force_reindex: bool = False):
    """
    1. Ensures SQL database is initialized.
    2. Seeds demo meetings if not already present.
    3. Generates embeddings dynamically and indexes all meetings into ChromaDB.
    """
    logger.info("Initializing relational database...")
    init_db()
    db = SessionLocal()

    try:
        for s_data in SAMPLE_MEETINGS:
            existing = db.query(Meeting).filter(Meeting.filename == s_data["filename"]).first()
            if not existing:
                logger.info(f"Seeding meeting: {s_data['filename']}")
                
                # Participants
                part_objs = []
                for p_name in s_data["participants"]:
                    p = db.query(Participant).filter(Participant.name == p_name).first()
                    if not p:
                        p = Participant(name=p_name)
                        db.add(p)
                    part_objs.append(p)
                db.commit()

                meeting = Meeting(
                    filename=s_data["filename"],
                    transcript=s_data["transcript"],
                    summary=s_data["summary"],
                )
                meeting.participants.extend(part_objs)
                db.add(meeting)
                db.commit()
                db.refresh(meeting)

                for ai in s_data["action_items"]:
                    p_id = None
                    if ai.get("assigned_participant"):
                        p = db.query(Participant).filter(Participant.name == ai["assigned_participant"]).first()
                        if p:
                            p_id = p.id
                    action = ActionItem(
                        meeting_id=meeting.id,
                        participant_id=p_id,
                        description=ai["description"],
                        deadline=ai.get("deadline"),
                        priority=ai.get("priority", "Medium"),
                        status=ai.get("status", "Pending")
                    )
                    db.add(action)

                for kp in s_data["key_points"]:
                    db.add(KeyPoint(meeting_id=meeting.id, point=kp))

                for kd in s_data["key_decisions"]:
                    db.add(KeyDecision(meeting_id=meeting.id, decision=kd))

                db.commit()
                logger.info(f"Created SQL record for meeting {meeting.id}: {s_data['filename']}")
            else:
                logger.info(f"Meeting already exists in SQL: {s_data['filename']}")
    finally:
        db.close()

    # Verify Knowledge Repository
    verify_rep = verify_knowledge_repository()
    logger.info(f"Knowledge Repository Verification: {verify_rep['status']} - {verify_rep['total_meetings']} meetings found")

    # Index into ChromaDB
    logger.info("Generating embeddings and syncing with ChromaDB vector store...")
    all_meetings = get_all_meetings()
    total_vectors_indexed = 0

    for m in all_meetings:
        units_with_embeddings = generate_meeting_embeddings(m)
        count = insert_meeting_vectors(m["id"], units_with_embeddings)
        total_vectors_indexed += count
        logger.info(f"Indexed Meeting ID {m['id']} ('{m['filename']}'): {count} vectors generated and stored.")

    stats = get_vector_store_stats()
    logger.info(f"Vector Database Sync Complete! Total vectors in store: {stats['total_vectors']}")
    logger.info(f"Vector Breakdown: {stats['breakdown']}")
    return stats

if __name__ == "__main__":
    seed_database_and_vector_store()
