from src.scoring import JobFitScorer
from src.skills import load_skills_db

def test_basic_scoring():
    scorer = JobFitScorer()
    skills_db = {
        "ml": ["python", "pytorch", "tensorflow", "scikit-learn"],
        "data": ["sql"]
    }
    resume = "Experienced ML Engineer with 3 years. Python, PyTorch, SQL."
    jd = "Looking for ML Engineer with 2+ years experience. Skills: Python, TensorFlow, SQL."
    result = scorer.score(resume, jd, skills_db=skills_db)
    assert "overall_score" in result
    assert "semantic" in result
    assert "coverage" in result
    assert "matched_skills" in result
    assert "missing_skills" in result
    print("Test passed. Overall score:", result["overall_score"])

if __name__ == "__main__":
    test_basic_scoring()
