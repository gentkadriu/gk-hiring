import json
import re
from typing import List, Dict, Set, Tuple
from rapidfuzz import fuzz
from collections import Counter
try:
    from nltk.corpus import stopwords  # type: ignore
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = {
        "a","an","and","the","to","of","in","for","on","with","at","by","from","as",
        "is","are","was","were","be","been","being","this","that","these","those","it",
        "or","not","but","if","then","so","than","too","very","can","will","shall","may",
        "i","you","he","she","we","they","them","us","our","your","their","my"
    }

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#-]{1,}")

def load_skills_db(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text.lower())]

def extract_skills(text: str, skills_db: Dict[str, List[str]], threshold: int = 90) -> Set[str]:
    tokens = set(tokenize(text))
    found: Set[str] = set()
    for group, skills in skills_db.items():
        for skill in skills:
            s = skill.lower()
            if s in tokens:
                found.add(s)
            else:
                for t in tokens:
                    if fuzz.token_set_ratio(s, t) >= threshold:
                        found.add(s)
                        break
    return found

def extract_keywords(text: str, top_n: int = 20, min_len: int = 2) -> List[Tuple[str, int]]:
    tokens = [t for t in tokenize(text) if len(t) >= min_len and t not in _STOPWORDS]
    counts = Counter(tokens)
    return counts.most_common(top_n)
