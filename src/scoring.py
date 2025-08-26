from typing import Dict, List, Set, Optional
from sentence_transformers import SentenceTransformer, util
from .skills import extract_skills, extract_keywords
import numpy as np
import re

def estimate_years_experience(text: str) -> float:
    yrs = []
    for m in re.finditer(r"(\d{1,2})\s*\+?\s*years", text.lower()):
        try:
            yrs.append(float(m.group(1)))
        except ValueError:
            pass
    return float(np.median(yrs)) if yrs else 0.0

class JobFitScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 weights: Dict[str, float] = None):
        self.model = SentenceTransformer(model_name)
        self.weights = weights or {"semantic": 0.6, "coverage": 0.3, "experience": 0.1}

    def score(self, resume_text: str, jd_text: str,
              skills_db: Optional[Dict[str, List[str]]] = None,
              weights: Optional[Dict[str, float]] = None) -> Dict:
        resume_text = resume_text.strip()
        jd_text = jd_text.strip()
        emb = self.model.encode([resume_text, jd_text], convert_to_tensor=True, normalize_embeddings=True)
        sim = util.cos_sim(emb[0], emb[1]).item()
        matched_skills: Set[str] = set()
        missing_skills: Set[str] = set()
        coverage = 0.0
        jd_keywords = extract_keywords(jd_text, top_n=25)
        resume_keywords = extract_keywords(resume_text, top_n=50)
        jd_kw_set = {k for k, _ in jd_keywords}
        resume_kw_set = {k for k, _ in resume_keywords}
        keyword_hits = jd_kw_set.intersection(resume_kw_set)
        keyword_coverage = len(keyword_hits) / max(1, len(jd_kw_set))
        if skills_db:
            resume_sk = extract_skills(resume_text, skills_db)
            jd_sk = extract_skills(jd_text, skills_db)
            matched_skills = resume_sk.intersection(jd_sk)
            missing_skills = jd_sk.difference(resume_sk)
            coverage = len(matched_skills) / max(1, len(jd_sk))
        yrs_resume = estimate_years_experience(resume_text)
        yrs_jd = estimate_years_experience(jd_text)
        if yrs_jd == 0:
            exp_score = 1.0
        else:
            exp_score = min(1.0, yrs_resume / max(0.5, yrs_jd))
        w = self.weights if weights is None else weights
        if "keywords" not in w:
            w = {**w, "keywords": 0.1}
            scale = (1.0 - w["keywords"]) / (w["semantic"] + w["coverage"] + w["experience"])
            w = {
                "semantic": w["semantic"] * scale,
                "coverage": w["coverage"] * scale,
                "experience": w["experience"] * scale,
                "keywords": w["keywords"],
            }
        overall = (w["semantic"] * sim +
                   w["coverage"] * coverage +
                   w["experience"] * exp_score +
                   w["keywords"] * keyword_coverage) * 100.0
        suggestions = []
        if missing_skills:
            suggestions.append("Add evidence/examples for: " + ", ".join(sorted(missing_skills)))
        if sim < 0.65:
            suggestions.append("Align phrasing with the JD. Mirror key role terms and project outcomes.")
        if coverage < 0.7 and skills_db:
            suggestions.append("Add sections for Tools/Frameworks and Certifications to boost coverage.")
        if exp_score < 0.8 and yrs_jd > 0:
            suggestions.append(f"If you have relevant experience not listed, quantify it (target: {yrs_jd:.0f}+ years).")
        if keyword_coverage < 0.6:
            suggestions.append("Include key JD terms verbatim where truthful to improve ATS keyword match.")
        suggestions.append("Rewrite bullets in STAR format: Situation, Task, Action, Result with metrics.")
        return {
            "overall_score": float(overall),
            "semantic": float(sim),
            "coverage": float(coverage),
            "experience": float(exp_score),
            "keyword_coverage": float(keyword_coverage),
            "jd_keywords": [k for k, _ in jd_keywords],
            "resume_keywords": [k for k, _ in resume_keywords],
            "keyword_hits": sorted(keyword_hits),
            "matched_skills": sorted(matched_skills),
            "missing_skills": sorted(missing_skills),
            "years_resume": yrs_resume,
            "years_jd": yrs_jd,
            "suggestions": suggestions,
        }
