# Configuration and settings for Milestone 4
import os
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Auto-load .env
for env_path in [
    os.path.join(BASE_DIR, ".env"),
    os.path.join(ROOT_DIR, ".env"),
    os.path.join(ROOT_DIR, "Milestone_3", ".env"),
]:
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and v:
                            os.environ[k] = v
        except Exception:
            pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Database settings
DB_PATH = os.path.join(BASE_DIR, "meeting_intelligence.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# ChromaDB settings
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "meeting_knowledge_store_m4"

# API & Security settings
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", 8000))
JWT_SECRET = os.environ.get("JWT_SECRET", "synthai-super-secret-production-key-2026")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# SLA target
TARGET_SLA_SECONDS = 3.0

# Supported upload formats
SUPPORTED_AUDIO_FORMATS: List[str] = [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]
SUPPORTED_VIDEO_FORMATS: List[str] = [".mp4", ".mov", ".mkv", ".avi"]
MAX_FILE_SIZE_MB = 100
