#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect TMDB person search results for actors and directors, matched via known_for titles.

- Input:
  --path (default: ./data) where Excel files exist
  Sample mode:
    SAMPLED_Leading and Leading Ensemble Actor.xlsx
    SAMPLED_Director-Producer-Exec_Producer-Screenwriter_sampled.xlsx
  Full mode:
    Leading and Leading Ensemble Actor.xls
    Director-Producer-Exec_Producer-Screenwriter.xlsx

- Output:
  --out_dir (default: TMDB_data_collected)
    actor_TMDB_data.json
    director_TMDB_data.json
"""

import os
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------
# CLI & Env
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Collect TMDB data for film people.")
    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Directory containing Excel source files.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampled Excel files (SAMPLED_*.xlsx).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="TMDB API key (or set TMDB_API_KEY env).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="TMDB_data_collected",
        help="Directory to save JSON outputs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout seconds for TMDB requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between TMDB calls to be polite.",
    )
    return parser.parse_args()


def sampling_enabled(args) -> bool:
    """Check sampling flag from CLI or environment variable."""
    if args.sample:
        return True
    env_val = os.getenv("SAMPLE", "").strip().lower()
    return env_val in {"1", "true", "t", "yes", "y"}


# ---------------------------
# Data helpers
# ---------------------------
def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Fix string encoding from Latin-1 to UTF-8 for object columns."""
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(
            lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x
        )
    return df


def make_person_movie_pairs(df: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
    """Build mapping {person_odid: {'person_name': ..., 'movies': [display_name,...]}}."""
    grouped = (
        df.groupby(["person_odid", "person"])["display_name"]
        .apply(lambda x: list(set(x)))
        .reset_index()
        .groupby("person_odid")
        .apply(
            lambda g: {
                "person_name": g["person"].iloc[0],
                "movies": g["display_name"].iloc[0],
            }
        )
    )
    return grouped.to_dict()


# ---------------------------
# TMDB helpers
# ---------------------------
BASE_URL = "https://api.themoviedb.org/3"


def normalize_title(title: str) -> str:
    """Lowercase + strip non-alnum (space kept) to compare titles."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def compare_filmography(list_a: List[str], list_b: List[str]) -> List[str]:
    """Return intersection (original titles from list_a) based on normalized matching."""
    set1 = {normalize_title(t) for t in list_a if isinstance(t, str)}
    set2 = {normalize_title(t) for t in list_b if isinstance(t, str)}
    common_norm = set1 & set2
    return [
        t for t in list_a if isinstance(t, str) and normalize_title(t) in common_norm
    ]


def get_movies_from_query(single_result: Dict[str, Any]) -> List[str]:
    """Extract titles/names from known_for entries (movie or TV)."""
    out = []
    for item in single_result.get("known_for", []) or []:
        if isinstance(item, dict):
            if item.get("title"):
                out.append(item["title"])
            elif item.get("name"):
                out.append(item["name"])
    return out


def query_person_by_name(
    session: requests.Session, api_key: str, name: str, timeout: int = 10
) -> Dict[str, Any]:
    """Call TMDB /search/person with a name."""
    url = f"{BASE_URL}/search/person"
    params = {"api_key": api_key, "query": name}
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_safe_result(
    person_key, pair_dict, session, api_key, timeout: int = 10
) -> Optional[Dict[str, Any]]:
    """Choose the best TMDB person search result by matching known_for titles with given filmography."""
    name = pair_dict[person_key]["person_name"]
    try:
        search_res = query_person_by_name(session, api_key, name, timeout=timeout)
    except Exception:
        return None

    results = search_res.get("results", []) or []
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        target_movies = pair_dict[person_key]["movies"] or []
        for result in results:
            known_for_titles = get_movies_from_query(result)
            if compare_filmography(known_for_titles, target_movies):
                return result
    return None


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    do_sample = sampling_enabled(args)

    # Resolve API key
    api_key = args.api_key or os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TMDB API key not provided. Use --api_key or set TMDB_API_KEY env."
        )

    # Resolve input file paths
    base_path = args.path.rstrip("/")

    # Full filenames
    actor_full = "Leading and Leading Ensemble Actor.xls"
    director_full = "Director-Producer-Exec_Producer-Screenwriter.xlsx"

    # Sample filenames
    actor_sample = "SAMPLED_Leading and Leading Ensemble Actor.xlsx"
    director_sample = (
        "SAMPLED_Director-Producer-Exec_Producer-Screenwriter_sampled.xlsx"
    )

    if do_sample:
        actor_path = os.path.join(base_path, actor_sample)
        director_path = os.path.join(base_path, director_sample)
        print(
            f"[Sampling Mode] Loading sampled Excel files:\n  - {actor_path}\n  - {director_path}"
        )
    else:
        actor_path = os.path.join(base_path, actor_full)
        director_path = os.path.join(base_path, director_full)
        print(
            f"[Full Mode] Loading full Excel files:\n  - {actor_path}\n  - {director_path}"
        )

    # Validate inputs
    for p in (actor_path, director_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input not found: {os.path.abspath(p)}")

    # Read Excels
    print("Reading Excel files ...")
    actor_df = pd.read_excel(actor_path)
    director_df = pd.read_excel(director_path)

    actor_df = fix_encoding(actor_df)
    director_df = fix_encoding(director_df)

    # Build person→movies pairs
    print("Building person/movie pairs ...")
    actor_pairs = make_person_movie_pairs(actor_df)
    director_pairs = make_person_movie_pairs(director_df)

    # Prepare output dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    actor_out = os.path.join(out_dir, "actor_TMDB_data.json")
    director_out = os.path.join(out_dir, "director_TMDB_data.json")

    # HTTP session
    session = requests.Session()
    session.headers.update({"User-Agent": "film-research/1.0"})

    # Collect actor results
    print("Querying TMDB for ACTORS ...")
    actor_collected = {}
    for key in tqdm(actor_pairs.keys()):
        actor_collected[key] = dict(actor_pairs[key])
        actor_collected[key]["detail"] = get_safe_result(
            key, actor_pairs, session=session, api_key=api_key, timeout=args.timeout
        )
        time.sleep(args.sleep)

    with open(actor_out, "w", encoding="utf-8") as fp:
        json.dump(actor_collected, fp, ensure_ascii=False)
    print(f"Saved: {os.path.abspath(actor_out)}")

    # Collect director results
    print("Querying TMDB for DIRECTORS ...")
    director_collected = {}
    for key in tqdm(director_pairs.keys()):
        director_collected[key] = dict(director_pairs[key])
        director_collected[key]["detail"] = get_safe_result(
            key, director_pairs, session=session, api_key=api_key, timeout=args.timeout
        )
        time.sleep(args.sleep)

    with open(director_out, "w", encoding="utf-8") as fp:
        json.dump(director_collected, fp, ensure_ascii=False)

    print(f"Saved: {os.path.abspath(director_out)}")

    print("\n--- Summary ---")
    print(
        {
            "mode": "sample" if do_sample else "full",
            "actor_pairs": len(actor_pairs),
            "director_pairs": len(director_pairs),
            "actor_out": os.path.abspath(actor_out),
            "director_out": os.path.abspath(director_out),
        }
    )


if __name__ == "__main__":
    main()
