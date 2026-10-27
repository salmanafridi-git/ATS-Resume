# 📄 ATS Resume Checker

A Streamlit app that scores your resume for Applicant Tracking Systems (ATS) and gives
specific suggestions to improve it, powered by Google Gemini Flash.

## Features
- Upload a resume in **PDF** or **DOCX** format
- Overall **ATS score (0-100)** with a 5-part breakdown
- Optional **job description** for a job-match score and missing keywords
- Strengths, weaknesses, and prioritized improvements with example rewrites
- Download the report as JSON

## Setup (local)

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide your API key, either way:
   - Paste it into the app's sidebar, **or**
   - Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
4. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (do **not** commit `.streamlit/secrets.toml`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- Scanned/image-only resumes can't be read (ATS systems can't read them either).
- The app uses `gemini-3.8-flash` and automatically falls back to `gemini-flash-latest` if a model is retired. To force a specific model, add `GEMINI_MODEL = "model-name"` to your Streamlit secrets.
- Scores are AI estimates, not the output of any real ATS.

## Files
- `app.py` – the Streamlit app
- `requirements.txt` – dependencies
- `README.md` – this file
