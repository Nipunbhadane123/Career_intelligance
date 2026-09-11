# AI Meeting Synthesizer

This project is a powerful AI-driven application that automates the transcription, speaker diarization, and summarization of meeting recordings. Built with Streamlit, it provides a user-friendly interface to upload audio/video files, view a structured report, and interactively chat with an AI assistant about the meeting's content.

## 🚀 Features

- **Audio Transcription:** Uses OpenAI's **Whisper** (`base` model) to accurately transcribe audio or video recordings.
- **Speaker Diarization:** Identifies who spoke when using **Pyannote** (`speaker-diarization-3.1`), assigning speaker labels to the transcribed text.
- **Automated Summarization:** Utilizes Google's **Gemini 3.7 Flash** to generate a professional, structured meeting report comprising:
  - Executive Summary
  - Key Discussion Points
  - Action Items & Owners
- **RAG-powered Chat Assistant:** Embeds chunks of the transcript into a local **ChromaDB** vector database using `sentence-transformers` (`all-MiniLM-L6-v2`). This allows users to ask questions about the meeting and receive accurate, context-aware answers from Gemini.
- **Export Options:** Download the final structured report as a **PDF** or **Word (DOCX)** document, or download the raw transcript as a `.txt` file.

## 📁 Project Structure

```text
meeting_synthesizer/
│
├── app.py                # Main Streamlit application containing the entire workflow and UI
├── requirements.txt      # Python dependencies required to run the app
├── ffmpeg.exe            # Executable for media file processing (needed by Whisper on Windows)
└── transcript.txt        # Sample output transcript file
```

## 🛠️ Tech Stack

- **Frontend / Framework:** [Streamlit](https://streamlit.io/)
- **Transcription:** [OpenAI Whisper](https://github.com/openai/whisper)
- **Speaker Diarization:** [Pyannote Audio](https://github.com/pyannote/pyannote-audio) & Hugging Face
- **LLM / Generative AI:** [Google Gemini API](https://ai.google.dev/) (`gemini-3.7-flash`)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** `sentence-transformers`
- **Document Generation:** `xhtml2pdf` (PDF), `python-docx` (Word)

## ⚙️ Step-by-Step Implementation Workflow

1. **Environment Setup & Workarounds:** 
   - The app initializes Streamlit. 
   - It runs workarounds for Windows to load `c10.dll` (to bypass a PyTorch initialization bug) and ensures `ffmpeg.exe` is copied to the local directory for Whisper to process media files correctly.
   
2. **API Configuration:**
   - The sidebar prompts the user for a **Gemini API Key** and a **Hugging Face Token** (required to use Pyannote).

3. **Model Caching:**
   - Deep learning models (Whisper, Pyannote pipeline, SentenceTransformers) and the ChromaDB client are loaded and cached using `@st.cache_resource` to avoid reloading them on every UI interaction.

4. **File Upload & Processing:**
   - Users upload an audio or video file (`mp3`, `wav`, `m4a`, `mp4`).
   - The file is saved temporarily. Whisper extracts the audio track and transcribes the speech into text segments.
   - Pyannote processes the audio tensor to generate a diarization timeline.

5. **Transcript Merging:**
   - The `merge_transcription_and_diarization` function aligns Whisper's text segments with Pyannote's speaker timeline to create a final, timestamped script indicating *who* said *what*.

6. **Indexing for RAG (Retrieval-Augmented Generation):**
   - The final transcript is split into smaller chunks.
   - These chunks are converted into vector embeddings via `sentence-transformers` and stored in a new ChromaDB collection named `meeting_chunks`.

7. **Report Generation:**
   - The full transcript is sent to Google's **Gemini API** with a prompt instructing it to create an Executive Summary, Key Discussion Points, and Action Items.

8. **User Interface (Tabs):**
   - **Tab 1 (Meeting Transcript):** Displays the generated report, allows exporting to PDF/DOCX, and shows the full raw transcript.
   - **Tab 2 (Chat with AI Assistant):** A chat interface where user queries are vectorized, searched against the ChromaDB collection to retrieve the most relevant transcript chunks, and sent to Gemini to formulate an accurate answer based purely on the meeting context.

## 💻 How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

3. **Use the App:**
   - Enter your API keys in the sidebar.
   - Upload a meeting recording.
   - Wait for the pipeline to finish processing.
   - Review the report, download it, or chat with your AI assistant!
