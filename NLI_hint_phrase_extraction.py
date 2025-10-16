#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Wikipedia bio CSVs (actors/directors) to extract race/ethnicity hint phrases.

- Input: actors_bio.csv, directors_bio.csv from a given folder (default: wiki_bio_collected/)
- Output: *_hint_phrases_processed_final_merged.csv in an output folder (default: wiki_bio_collected/processed/)
- Comments are in English as requested.
"""

import argparse
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
from typing import List, Optional, Tuple, Any


# ---------------------------
# CLI argument configuration
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process wiki bios to extract race-related hint phrases."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="wiki_bio_collected",
        help="Input directory containing actors_bio.csv and directors_bio.csv",
    )
    parser.add_argument(
        "--actors_file",
        type=str,
        default="actors_bio.csv",
        help="Actors CSV filename (inside in_dir)",
    )
    parser.add_argument(
        "--directors_file",
        type=str,
        default="directors_bio.csv",
        help="Directors CSV filename (inside in_dir)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory; default is <in_dir>/processed",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout (seconds) for category scraping",
    )
    return parser.parse_args()


# ---------------------------
# Regex patterns & constants
# ---------------------------
STEP1_KEYWORDS = [
    "descent",
    "descendant",
    "heritage",
    "ancestry",
    "ethnicity",
    "parents",
    "mother",
    "father",
    "immigrant",
    "family background",
    "roots",
]

STEP2_SOFT_KEYWORDS = [
    r"born to .* family",
    r"raised in .* neighborhood",
    r"from .* background",
    r"has .* heritage",
    r"of .* origin",
    r"with .* roots",
]

PHRASE_PATTERN = re.compile(
    "|".join(STEP1_KEYWORDS + STEP2_SOFT_KEYWORDS), re.IGNORECASE
)

KNOWN_RACE_WORDS = {
    "Asian",
    "Black",
    "African",
    "Latino",
    "Latina",
    "Hispanic",
    "White",
    "Arab",
    "Indian",
    "Jewish",
    "Korean",
    "Japanese",
    "Chinese",
    "Vietnamese",
    "Filipino",
    "Native",
    "Mexican",
    "Persian",
    "Pakistani",
    "Bangladeshi",
    "Thai",
    "Indigenous",
    "English",
    "Scottish",
    "Welsh",
    "Irish",
    "German",
    "Italian",
    "Spanish",
    "Swedish",
    "Norwegian",
    "Danish",
    "Dutch",
    "French",
    "Portuguese",
}

DESCENT_PATTERNS = [
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?) descent\b",
    r"\bof ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?) descent\b",
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?) ancestry\b",
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?) heritage\b",
    r"\bparents.*?\bfrom\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b",
    r"\bparents.*?\bare\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b",
]

AMERICAN_NATIONALITY_RGX = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*[-\s]?American\b"
)


# ---------------------------
# Text extraction helpers
# ---------------------------
def extract_hint_phrases(text: Any) -> List[str]:
    """Return sentences likely containing race/heritage hints."""
    if not isinstance(text, str):
        return []
    sentences = [s.strip() for s in text.split(".") if len(s) > 20]
    return [s for s in sentences if PHRASE_PATTERN.search(s)]


def extract_language_ethnicity(text: Any) -> Optional[str]:
    """Extract language/ethnicity token from wiki summary artifacts if present."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\(\s*;\s*(\w+):\s*\[", text)
    return match.group(1) if match else None


def extract_birthplace_info(text: Any) -> Optional[List[str]]:
    """Extract simple birthplace/immigrant signals."""
    if not isinstance(text, str):
        return None
    pattern = re.compile(
        r"(born in [^.,;\n]+|raised in [^.,;\n]+|immigrant|family.*from [^.,;\n]+)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    return matches if matches else None


def extract_country_hint(text: Any) -> Optional[str]:
    """Extract nationality-like token from first sentence (heuristic)."""
    if not isinstance(text, str):
        return None
    first_sentence = text.split(".")[0]
    match = re.search(
        r"\b(?:is|was)\s+an?\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", first_sentence
    )
    return match.group(1) if match else None


def extract_race_hint_fields(text: Any) -> dict:
    """Aggregate multiple heuristics into a structured dict."""
    result = {}
    phrases = extract_hint_phrases(text)
    if phrases:
        result["race_hint_phrases"] = phrases

    lang_eth = extract_language_ethnicity(text)
    if lang_eth:
        result["language_ethnicity"] = lang_eth

    birthplace = extract_birthplace_info(text)
    if birthplace:
        result["birthplace_signal"] = birthplace

    country = extract_country_hint(text)
    if country:
        result["country_hint"] = country

    if result:
        result["race_hints_all"] = {
            "phrases": result.get("race_hint_phrases"),
            "language": result.get("language_ethnicity"),
            "birthplace": result.get("birthplace_signal"),
            "country": result.get("country_hint"),
        }
    return result


# ---------------------------
# Wikipedia category scraping
# ---------------------------
def extract_wikipedia_categories(
    url: str, session: requests.Session, timeout: int = 10
) -> List[str]:
    """Fetch category links from a Wikipedia page."""
    try:
        if not isinstance(url, str) or not url:
            return []
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        cat_div = soup.find("div", id="catlinks")
        if not cat_div:
            return []
        return [a.text.strip() for a in cat_div.select("ul li a")]
    except Exception as e:
        return [f"ERROR: {e}"]


def extract_descent_and_american_phrases(text: str, nlp) -> List[str]:
    """Extract nationality/heritage phrases from a category-like string."""
    if not isinstance(text, str):
        return []
    phrases = []
    for pattern in DESCENT_PATTERNS:
        phrases.extend(re.findall(pattern, text))

    # Capture 'X-American' style categories (e.g., 'Korean-American')
    american_matches = AMERICAN_NATIONALITY_RGX.findall(text)
    for match in american_matches:
        doc = nlp(match)
        for ent in doc.ents:
            if ent.label_ in {"NORP", "ETHNIC_GROUP", "ETHNICITY"}:
                phrases.append(f"{match} American")

    return list(set(phrases))


def add_race_hint_from_categories(
    df: pd.DataFrame, url_col: str, nlp, timeout: int = 10
) -> pd.DataFrame:
    """Add category-derived race hints to the DataFrame."""
    wiki_categories, race_hints = [], []
    session = requests.Session()

    for url in tqdm(df[url_col], desc="Processing Wikipedia Categories"):
        cats = extract_wikipedia_categories(url, session=session, timeout=timeout)
        wiki_categories.append(cats)

        hint = []
        if isinstance(cats, list):
            for cat in cats:
                hint.extend(extract_descent_and_american_phrases(cat, nlp))
        race_hints.append(list(set(hint)))

    df["wiki_categories"] = wiki_categories
    df["wiki_categories_racehint"] = race_hints
    return df


# ---------------------------
# Reliable phrase extraction
# ---------------------------
def is_race_by_ner(word: str, nlp) -> bool:
    """Use spaCy NER to judge whether a token looks like a nationality/ethnic group."""
    doc = nlp(word)
    for ent in doc.ents:
        if ent.label_ in {"NORP", "ETHNICITY", "ETHNIC_GROUP"}:
            return True
    return False


def extract_reliable_phrases(text: Any, nlp) -> List[Tuple[str, List[str]]]:
    """Find reliable sentences and associated race keywords within them."""
    if not isinstance(text, str):
        return []

    text = text.strip()
    matched_keywords: List[str] = []

    # Pattern-based matching (descent/ancestry/heritage/parents from)
    for pattern in DESCENT_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            matched_keywords.extend(matches)

    # NER-supported patterns
    ner_patterns = [
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:mom|dad|parents)\b",
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*(?:-|\s)?American\b",
    ]
    for pattern in ner_patterns:
        candidates = re.findall(pattern, text)
        for c in candidates:
            if c in KNOWN_RACE_WORDS or is_race_by_ner(c, nlp):
                matched_keywords.append(c)

    if matched_keywords:
        return [(text, list(set(matched_keywords)))]
    return []


def reliable_phrases_and_keyword(phrase_list: Any, nlp) -> Tuple[List[str], List[str]]:
    """Split into (reliable_phrases, keywords) from a list of candidate phrases."""
    phrases, keywords = [], []
    if not isinstance(phrase_list, list):
        return [], []
    for sent in phrase_list:
        extracted = extract_reliable_phrases(sent, nlp)
        if extracted:
            p, k = extracted[0]
            phrases.append(p)
            keywords.extend(k)
    if not phrases:
        return [], []
    return list(set(phrases)), list(set(keywords))


def extract_race_keywords_from_sentences(sentences: Any, nlp) -> List[str]:
    """NER over a list of sentences to pull NORP/ETHNICITY entities."""
    race_keywords = []
    if not isinstance(sentences, list):
        return []
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ in {"NORP", "ETHNICITY", "ETHNIC_GROUP"}:
                race_keywords.append(ent.text.strip())
    return list(set(race_keywords))


# ---------------------------
# Merge helpers
# ---------------------------
def merge_and_deduplicate(*args) -> Optional[List[str]]:
    """Merge multiple lists and deduplicate."""
    merged: List[str] = []
    for lst in args:
        if isinstance(lst, list):
            merged.extend(lst)
    return list(set(merged)) if merged else None


def list_empty_to_none(x: Any) -> Any:
    """Convert [] to None to keep CSV cleaner."""
    return None if isinstance(x, list) and len(x) == 0 else x


# ---------------------------
# Frame-level pipeline
# ---------------------------
def process_frame(df: pd.DataFrame, nlp, timeout: int) -> pd.DataFrame:
    """Full pipeline per DataFrame (actors/directors)."""
    # Validate required columns
    required_cols = {"summary", "early_life", "url", "name", "movie", "title"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Combine summary + early life
    df["summary_earlyLife_combined"] = (
        df["summary"].fillna("") + " " + df["early_life"].fillna("")
    )

    # Extract hint field bundle
    extracted = df["summary_earlyLife_combined"].apply(extract_race_hint_fields)
    df["race_hint_phrases"] = extracted.apply(
        lambda x: x.get("race_hint_phrases") if x else None
    )
    df["language_ethnicity"] = extracted.apply(
        lambda x: x.get("language_ethnicity") if x else None
    )
    df["birthplace_signal"] = extracted.apply(
        lambda x: x.get("birthplace_signal") if x else None
    )
    df["country_hint"] = extracted.apply(lambda x: x.get("country_hint") if x else None)
    df["race_hints_all"] = extracted.apply(
        lambda x: x.get("race_hints_all") if x else None
    )

    # Categories → race hints
    df = add_race_hint_from_categories(df, url_col="url", nlp=nlp, timeout=timeout)

    # Reliable phrases / keywords from description phrases
    df["phrases"] = df["race_hints_all"].apply(
        lambda x: x.get("phrases") if isinstance(x, dict) else None
    )
    df["wiki_description_reliable_phrases"] = df["phrases"].apply(
        lambda x: reliable_phrases_and_keyword(x, nlp)[0] if x is not None else None
    )
    df["wiki_description_racehint"] = df["phrases"].apply(
        lambda x: reliable_phrases_and_keyword(x, nlp)[1] if x is not None else None
    )

    # Keep only analysis columns for next stage
    cols_for_next = [
        "name",
        "movie",
        "title",
        "summary",
        "wiki_categories_racehint",
        "wiki_description_racehint",
        "wiki_description_reliable_phrases",
        "country_hint",
        "language_ethnicity",
        "birthplace_signal",
    ]
    df_next = df[cols_for_next].copy()

    # NER over reliable phrases and birthplace signals
    df_next["NER_racehint"] = df_next["wiki_description_reliable_phrases"].apply(
        lambda x: (
            extract_race_keywords_from_sentences(x, nlp) if x is not None else None
        )
    )
    df_next["NER_bornhint"] = df_next["birthplace_signal"].apply(
        lambda x: (
            extract_race_keywords_from_sentences(x, nlp) if x is not None else None
        )
    )

    # Merge lists of hints to a single field
    merge_cols = [
        "wiki_categories_racehint",
        "wiki_description_racehint",
        "NER_racehint",
        "NER_bornhint",
    ]
    df_next["merged_race_keywords"] = df_next[merge_cols].apply(
        lambda row: merge_and_deduplicate(*row), axis=1
    )

    # Final column order
    final_cols = [
        "name",
        "movie",
        "title",
        "summary",
        "country_hint",
        "merged_race_keywords",
    ]
    df_final = df_next[final_cols].copy()

    # Tidy: [] → None
    df_final = df_final.applymap(list_empty_to_none)
    return df_final


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    in_dir = args.in_dir.rstrip("/")

    out_dir = args.out_dir or os.path.join(in_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)

    actors_path = os.path.join(in_dir, args.actors_file)
    directors_path = os.path.join(in_dir, args.directors_file)

    # Load spaCy once
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Load CSVs
    print("Reading input CSVs...")
    df_actor_success = pd.read_csv(actors_path)
    df_directors_success = pd.read_csv(directors_path)

    # Process
    print("Processing ACTORS...")
    df_actor_final = process_frame(df_actor_success, nlp=nlp, timeout=args.timeout)

    print("Processing DIRECTORS...")
    df_director_final = process_frame(
        df_directors_success, nlp=nlp, timeout=args.timeout
    )

    # Save
    actors_out = os.path.join(
        out_dir, "actors_bio_hint_phrases_processed_final_merged.csv"
    )
    directors_out = os.path.join(
        out_dir, "directors_bio_hint_phrases_processed_final_merged.csv"
    )

    df_actor_final.to_csv(actors_out, index=False)
    df_director_final.to_csv(directors_out, index=False)

    print("\n--- Summary ---")
    print(
        {
            "in_dir": in_dir,
            "out_dir": out_dir,
            "actors_rows_in": len(df_actor_success),
            "actors_rows_out": len(df_actor_final),
            "directors_rows_in": len(df_directors_success),
            "directors_rows_out": len(df_director_final),
        }
    )


if __name__ == "__main__":
    main()
