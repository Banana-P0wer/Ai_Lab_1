import argparse
import csv
import html
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd

URL_EMAIL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
APOSTROPHE_RE = re.compile(r"[’`´ʹʻ]")
REPEAT_CHAR_RE = re.compile(r"(?i)([a-z])\1{2,}")
NON_TEXT_RE = re.compile(r"[^a-z0-9_ \t\n\.\,$begin:math:text$$end:math:text$$begin:math:display$$end:math:display$\{\}\-\#\+\:\;\/\=\<\>\!\%\*\'\"]", re.IGNORECASE)

CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "weren't": "were not", "wasn't": "was not",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "what's": "what is", "who's": "who is", "let's": "let us"
}

def load_lexicon(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

def normalize_unicode(s: str) -> str:
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    return APOSTROPHE_RE.sub("'", s)

def expand_contractions(text: str) -> str:
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"(?i)\b{k}\b", v, text)
    return text

def clean_text(text: str, profane_words: set[str], prog_words: set[str]) -> str:
    if not isinstance(text, str):
        return ""
    t = normalize_unicode(text).strip().lower()
    t = expand_contractions(URL_EMAIL_RE.sub(" ", t))
    for word in profane_words:
        pattern = r"(?i)" + r"".join(f"{c}[^a-z0-9]{{0,2}}" for c in word)
        t = re.sub(pattern, word, t)
    t = NON_TEXT_RE.sub(" ", t)
    tokens = []
    for tok in re.split(r"(\s+)", t):
        if tok.isspace():
            tokens.append(tok)
            continue
        if any(c.isdigit() or c in "_./#-=%*()[]{}" for c in tok) or tok in prog_words:
            tokens.append(tok)
        else:
            tokens.append(REPEAT_CHAR_RE.sub(r"\1\1", tok))
    return MULTISPACE_RE.sub(" ", "".join(tokens)).strip()

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype={"message": "string", "is_toxic": "Int64"})
    if not {"message", "is_toxic"}.issubset(df.columns):
        raise ValueError(f"{path}: отсутствуют обязательные колонки 'message' и 'is_toxic'")
    return df

def preprocess(df: pd.DataFrame, name: str, profane: set[str], prog: set[str]) -> tuple[pd.DataFrame, dict]:
    orig = len(df)
    df["message"] = df["message"].fillna("").astype(str)
    df["message"] = df["message"].map(lambda x: clean_text(x, profane, prog))
    df = df[df["message"].str.len() > 0].dropna(subset=["is_toxic"]).drop_duplicates(subset=["message", "is_toxic"])
    df["is_toxic"] = df["is_toxic"].astype(int)
    stats = {
        "dataset": name,
        "orig_rows": orig,
        "final_rows": len(df),
        "class_0": int((df["is_toxic"] == 0).sum()),
        "class_1": int((df["is_toxic"] == 1).sum()),
    }
    return df, stats

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

def print_stats(stats: dict) -> None:
    total = stats["class_0"] + stats["class_1"]
    ratio = stats["class_1"] / total if total else 0
    print(f"[{stats['dataset']}] {stats['final_rows']} строк | токсичных: {ratio:.3f}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-in", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--test-in", type=Path, default=Path("data/raw/test.csv"))
    p.add_argument("--train-out", type=Path, default=Path("data/preprocessed/clean_train.csv"))
    p.add_argument("--test-out", type=Path, default=Path("data/preprocessed/clean_test.csv"))
    p.add_argument("--lexicons", type=Path, default=Path("data/lexicons"))
    return p.parse_args()

def process_dataset(in_path: Path, out_path: Path, name: str, profane: set[str], prog: set[str]) -> dict:
    df, stats = preprocess(load_csv(in_path), name, profane, prog)
    save_csv(df, out_path)
    print_stats(stats)
    return stats

def main() -> int:
    args = parse_args()
    profane = load_lexicon(args.lexicons / "profane-words.txt")
    prog = load_lexicon(args.lexicons / "programming_keywords.txt")
    print(f"Загружено {len(profane)} ругательных и {len(prog)} технических слов.")
    for path in [args.train_in, args.test_in]:
        if not path.exists():
            print(f"Ошибка: отсутствует {path}", file=sys.stderr)
            return 1
    datasets = [
        (args.train_in, args.train_out, "train"),
        (args.test_in, args.test_out, "test")
    ]
    for in_path, out_path, name in datasets:
        process_dataset(in_path, out_path, name, profane, prog)
    print("\nФайлы сохранены:\n ", args.train_out, "\n ", args.test_out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
