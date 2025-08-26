# GK Hiring – AI Resume Analyzer 

Analyze a resume against a job description with ATS-aware keyword coverage, semantic similarity, skill matching, and actionable suggestions. Built with Python and Streamlit.

## Highlights
- Upload resume/JD (PDF/DOCX) or paste text
- Transformer-based semantic similarity (Sentence-BERT)
- Skill extraction with fuzzy matching (configurable skills DB)
- ATS keyword insights (JD vs resume keywords, coverage %)
- Experience alignment (years mentioned in texts)
- Tunable weights (semantic, skills, experience, keywords)
- Export results as JSON or Markdown

## Quickstart

1) Clone and install

git clone https://github.com/gentkadriu/gk-hiring.git
cd gk-hiring
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt

2) Run the app

streamlit run app/app.py

## How to use
1) Provide a resume and a job description (upload file or paste text).
2) Optionally adjust scoring weights in the left sidebar.
3) Click Analyze to see:
   - Overall Fit, Semantic Similarity, Skill Coverage, Keyword Coverage, Experience Score
   - Matched vs Missing skills
   - ATS keyword insights (JD keywords, resume keywords, hits)
   - Actionable suggestions (includes STAR-format guidance)
4) Download the report as JSON or Markdown.

## Tech notes
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Keyword extraction: simple token frequency with stopword removal
- Skills DB: `data/skills_db.json` (includes AI/LLM, MLOps, evaluation, safety)

## Tests

python -m pytest -q
