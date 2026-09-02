<div align="center">

# 🧠 Career Intelligence

### *AI-Powered Meeting Synthesis & Career Intelligence Suite*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://careerintelligance-eedryyrvermedn5wsjyvdx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%20–%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Whisper](https://img.shields.io/badge/Speech-OpenAI%20Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)

<br/>

> **Transform unstructured meeting recordings into structured, actionable intelligence.**
> Upload a recording → get a speaker-labelled transcript, executive summary, action items,
> and an AI assistant you can chat with — all in seconds.

<br/>

</div>

---

## ✨ What's Inside

This repository is organized as a series of milestones, each shipping a standalone, production-ready AI application:

| Milestone | App | Description | Status |
|---|---|---|---|
| **Milestone 1** | 🧠 SynthAI — Meeting Intelligence | End-to-end meeting transcription, diarization, summarization & RAG-chat | ✅ Live |

---

## 🧠 Milestone 1 — SynthAI: Meeting Intelligence

> 📂 [`Milestone_1/`](Milestone_1/)
> 🔗 **[Try the live app →](https://careerintelligance-eedryyrvermedn5wsjyvdx.streamlit.app/)**

SynthAI is a full-stack AI pipeline that takes any meeting recording and turns it into a polished intelligence report — completely automated.

### 🔄 How the Pipeline Works

```
 Audio / Video File
        │
        ▼
 ┌─────────────────┐        ┌──────────────────────┐
 │  FFmpeg Extract │──────▶ │  Whisper Transcription│
 │  (audio track)  │        │  (speech-to-text)     │
 └─────────────────┘        └──────────┬───────────┘
                                        │
        ┌───────────────────────────────┤
        ▼                               ▼
 ┌─────────────────┐        ┌──────────────────────┐
 │    Pyannote     │        │   Transcript Merger   │
 │ (who spoke when)│──────▶ │  (Speaker A: "...")   │
 └─────────────────┘        └──────────┬───────────┘
                                        │
               ┌────────────────────────┤
               ▼                        ▼
  ┌───────────────────┐    ┌──────────────────────────┐
  │  Google Gemini    │    │  SentenceTransformers     │
  │  (Summarization,  │    │  + ChromaDB               │
  │  Action Items)    │    │  (RAG vector store)       │
  └────────┬──────────┘    └──────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │  📄 Executive Report  (PDF / DOCX)   │
  │  💬 Chat with the Meeting (RAG)      │
  └──────────────────────────────────────┘
```

### 🚀 Key Features

| Feature | Detail |
|---|---|
| 🎙️ **Speech-to-Text** | OpenAI Whisper `base` model — accurate, multi-lingual |
| 👥 **Speaker Diarization** | Pyannote 3.1 — identifies *who* said *what* and *when* |
| 📝 **Executive Report** | Gemini-generated summary, key decisions & action items |
| 💬 **RAG Chat Assistant** | Ask anything about the meeting; answers grounded in transcript |
| 📥 **Export** | Download as **PDF**, **Word (DOCX)**, or raw `.txt` |
| ⚡ **Model Caching** | Streamlit `@st.cache_resource` keeps models warm between runs |

### 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI / Framework** | [Streamlit](https://streamlit.io/) |
| **Transcription** | [OpenAI Whisper](https://github.com/openai/whisper) |
| **Diarization** | [Pyannote Audio 3.1](https://github.com/pyannote/pyannote-audio) |
| **LLM** | [Google Gemini Flash](https://ai.google.dev/) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Media Processing** | FFmpeg + MoviePy |
| **Export** | `xhtml2pdf`, `python-docx` |

---

## 🖥️ Local Setup

### Prerequisites

- Python **3.9 – 3.11**
- A [Google Gemini API Key](https://aistudio.google.com/) (free tier works)
- A [HuggingFace Access Token](https://huggingface.co/settings/tokens) (required for Pyannote)
- Accept model agreements on HuggingFace:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Quick Start

```bash
# 1. Clone
git clone https://github.com/Nipunbhadane123/Career_intelligance.git
cd Career_intelligance/Milestone_1

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Open **http://localhost:8501**, enter your API keys in the sidebar, upload a recording, and let SynthAI do the rest.

> [!WARNING]
> **PyTorch / CUDA**: If you have a GPU, install the matching CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt` for best performance.

> [!NOTE]
> **Windows users**: A bundled `ffmpeg.exe` is included in `Milestone_1/` for convenience. If you prefer a system-wide install, grab it from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your `PATH`.

---

## 🗂️ Repository Structure

```
Career_intelligance/
│
├── Milestone_1/
│   ├── app.py              ← Full Streamlit application
│   ├── requirements.txt    ← Python dependencies
│   ├── ffmpeg.exe          ← Bundled FFmpeg binary (Windows)
│   └── README.md           ← Milestone-specific deep-dive docs
│
├── packages.txt            ← System packages for Streamlit Cloud
├── LICENSE                 ← MIT License
└── README.md               ← You are here
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Made with ❤️ by **Nipun Bhadane**

</div>
