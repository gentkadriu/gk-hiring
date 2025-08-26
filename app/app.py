import json
import io
import streamlit as st
from src.resume_parser import extract_text_from_any
from src.scoring import JobFitScorer
from src.skills import load_skills_db
from src.io_utils import save_report

st.set_page_config(page_title="gk-hiring", layout="wide")
st.title("gk-hiring - 🧠 AI Resume Analyzer")
st.caption("ATS-aware semantic analysis with skill coverage, keyword insights, and actionable guidance.")

@st.cache_resource
def get_scorer():
    return JobFitScorer(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_skills_db():
    return load_skills_db("data/skills_db.json")

with st.sidebar:
    st.subheader("Scoring Weights")
    st.caption("Choose a preset or fine-tune. Hover the ⓘ icons for help.")

    preset = st.radio(
        "Preset",
        ["Balanced", "ATS-focused", "Skills-focused", "Experience-focused", "Custom"],
        index=0,
        help="Quickly switch weighting strategies. Choose Custom to use the sliders.")

    w_sem = st.slider(
        "Semantic",
        0.0, 1.0, 0.6, 0.05,
        help="How closely the resume meaning matches the JD using transformer embeddings.")
    w_cov = st.slider(
        "Skill Coverage",
        0.0, 1.0, 0.3, 0.05,
        help="Share of JD skills found in the resume (matched vs required).")
    w_exp = st.slider(
        "Experience",
        0.0, 1.0, 0.1, 0.05,
        help="How well years of experience in the resume meet the JD's stated years.")
    w_key = st.slider(
        "Keywords (ATS)",
        0.0, 1.0, 0.1, 0.05,
        help="Presence of JD keywords in the resume to improve ATS screening.")

    if preset != "Custom":
        if preset == "Balanced":
            base = {"semantic": 0.55, "coverage": 0.25, "experience": 0.10, "keywords": 0.10}
        elif preset == "ATS-focused":
            base = {"semantic": 0.40, "coverage": 0.20, "experience": 0.10, "keywords": 0.30}
        elif preset == "Skills-focused":
            base = {"semantic": 0.45, "coverage": 0.40, "experience": 0.05, "keywords": 0.10}
        else:
            base = {"semantic": 0.45, "coverage": 0.15, "experience": 0.30, "keywords": 0.10}
        weights = base
    else:
        total_w = max(1e-6, (w_sem + w_cov + w_exp + w_key))
        weights = {
            "semantic": w_sem / total_w,
            "coverage": w_cov / total_w,
            "experience": w_exp / total_w,
            "keywords": w_key / total_w,
        }

left, right = st.columns(2)

with left:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload resume (PDF/DOCX) or paste below", type=["pdf","docx"], key="resume")
    resume_text = st.text_area("…or paste resume text", height=220)

with right:
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload JD (PDF/DOCX) or paste below", type=["pdf","docx"], key="jd")
    jd_text = st.text_area("…or paste JD text", height=220)

if st.button("Analyze", type="primary"):
    if not (resume_file or resume_text) or not (jd_file or jd_text):
        st.warning("Please provide both a resume and a job description.")
        st.stop()

    with st.spinner("Reading files and extracting text…"):
        resume_raw = extract_text_from_any(resume_file) if resume_file else resume_text
        jd_raw = extract_text_from_any(jd_file) if jd_file else jd_text

    if not resume_raw.strip() or not jd_raw.strip():
        st.error("Could not read text from one of the inputs. Try pasting text.")
        st.stop()

    scorer = get_scorer()
    skills_db = get_skills_db()

    with st.spinner("Scoring resume vs JD…"):
        result = scorer.score(resume_raw, jd_raw, skills_db=skills_db, weights=weights)

    st.success("Analysis complete.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Fit", f"{result['overall_score']:.1f}/100")
    c2.metric("Semantic Similarity", f"{result['semantic']*100:.1f}%")
    c3.metric("Skill Coverage", f"{result['coverage']*100:.1f}%")

    c4, c5 = st.columns(2)
    c4.metric("Keyword Coverage (ATS)", f"{result['keyword_coverage']*100:.1f}%")
    c5.metric("Experience Score", f"{result['experience']*100:.1f}%")

    st.subheader("Matched & Missing Skills")
    left2, right2 = st.columns(2)
    with left2:
        st.markdown("**Matched skills**")
        st.write(sorted(result["matched_skills"]))
    with right2:
        st.markdown("**Missing skills**")
        st.write(sorted(result["missing_skills"]))

    st.subheader("Suggestions")
    st.write("\n".join(result["suggestions"]))

    st.subheader("Keyword Insights (ATS)")
    st.markdown("**JD Keywords**")
    st.write(result.get("jd_keywords", []))
    st.markdown("**Resume Keywords**")
    st.write(result.get("resume_keywords", []))
    st.markdown("**Hits (present in both)**")
    st.write(result.get("keyword_hits", []))

    st.subheader("Report JSON")
    st.json(result)

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, indent=2).encode("utf-8"),
        file_name="resume_analysis.json",
        mime="application/json"
    )

    md = io.StringIO()
    md.write(f"# AI Resume Analyzer Report\n\n")
    md.write(f"**Overall Fit:** {result['overall_score']:.1f}/100\n\n")
    md.write(f"**Semantic Similarity:** {result['semantic']*100:.1f}%\n\n")
    md.write(f"**Skill Coverage:** {result['coverage']*100:.1f}%\n\n")
    md.write("## Matched Skills\n\n" + ", ".join(sorted(result['matched_skills'])) + "\n\n")
    md.write("## Missing Skills\n\n" + ", ".join(sorted(result['missing_skills'])) + "\n\n")
    md.write("## Suggestions\n\n" + "\n".join(result['suggestions']) + "\n")
    md.write("## ATS Keyword Insights\n\n")
    md.write("JD: " + ", ".join(result.get('jd_keywords', [])) + "\n\n")
    md.write("Resume: " + ", ".join(result.get('resume_keywords', [])) + "\n\n")
    md.write("Hits: " + ", ".join(result.get('keyword_hits', [])) + "\n")
    st.download_button("Download report (Markdown)", md.getvalue(), file_name="resume_analysis.md")
