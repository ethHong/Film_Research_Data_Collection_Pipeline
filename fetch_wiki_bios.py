#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import wikipedia
from tqdm import tqdm
from difflib import get_close_matches
import argparse


# ---------------------------
# CLI argument configuration
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Wikipedia bio scraper for film people."
    )
    parser.add_argument("--path", type=str, default=".", help="Path to Excel files.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling mode (load pre-sampled files).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples (ignored when sample mode loads files).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (if used)."
    )
    return parser.parse_args()


def sampling_enabled(args):
    """Check sampling flag from CLI or environment variable."""
    if args.sample:
        return True
    env_val = os.getenv("SAMPLE", "").strip().lower()
    return env_val in {"1", "true", "t", "yes", "y"}


# ---------------------------
# Utility functions
# ---------------------------
def normalize_name(name):
    """Normalize a name by removing punctuation and extra spaces."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def fetch_bio_with_full_content(name, movie_hint=None, similarity_threshold=0.8):
    """Search Wikipedia for a person and fetch full content and early life info."""
    try:
        results = wikipedia.search(name)
        if not results:
            return {"status": "no_results", "query": name}

        name_clean = normalize_name(name)
        close_titles = get_close_matches(
            name_clean,
            [normalize_name(t) for t in results],
            n=5,
            cutoff=similarity_threshold,
        )

        for title in results:
            title_clean = normalize_name(title)
            if title_clean not in close_titles:
                continue
            try:
                page = wikipedia.page(title, auto_suggest=False)
                full_content = page.content
                early_life = (
                    page.section("Early life")
                    or page.section("Biography")
                    or page.section("Early life and education")
                )
                # If a movie title is provided, ensure it's mentioned in the content
                if movie_hint and movie_hint.lower() not in full_content.lower():
                    continue
                return {
                    "status": "success",
                    "query": name,
                    "title": page.title,
                    "url": page.url,
                    "summary": page.summary,
                    "early_life": early_life,
                    "content_snippet": full_content,
                }
            except (
                wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
            ):
                continue
        return {"status": "no_valid_person_page", "query": name}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def fix_encoding(df):
    """Fix string encoding from Latin-1 to UTF-8 for text columns."""
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(
            lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x
        )
    return df


# ---------------------------
# Main workflow
# ---------------------------
def main():
    args = parse_args()
    do_sample = sampling_enabled(args)
    N = max(1, args.n)
    base_path = args.path.rstrip("/")

    # Default (full) filenames
    actor_filename_full = "Leading and Leading Ensemble Actor.xls"
    director_filename_full = "Director-Producer-Exec_Producer-Screenwriter.xlsx"

    # Sampled filenames (xlsx as you specified)
    actor_filename_sample = "SAMPLED_Leading and Leading Ensemble Actor.xlsx"
    director_filename_sample = (
        "SAMPLED_Director-Producer-Exec_Producer-Screenwriter_sampled.xlsx"
    )

    # Choose filenames based on sampling toggle
    if do_sample:
        actor_path = os.path.join(base_path, actor_filename_sample)
        director_path = os.path.join(base_path, director_filename_sample)
        print(
            f"[Sampling Mode] Loading sampled files:\n- {actor_path}\n- {director_path}"
        )
    else:
        actor_path = os.path.join(base_path, actor_filename_full)
        director_path = os.path.join(base_path, director_filename_full)
        print(f"[Full Mode] Loading full files:\n- {actor_path}\n- {director_path}")

    # Validate existence
    for p in [actor_path, director_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Excel not found: {os.path.abspath(p)}")

    # Load Excel files (engine auto-detected by pandas)
    print("Loading Excel files...")
    director = pd.read_excel(director_path)
    actor = pd.read_excel(actor_path)

    director = fix_encoding(director)
    actor = fix_encoding(actor)

    # NOTE: When sampling mode is enabled, we DO NOT sample rows here.
    if do_sample:
        print("Sampling mode -> Using pre-sampled files; row sampling is skipped.")
    else:
        # Optional: keep legacy row-sampling when not in sampling mode
        if N and N < len(director):
            director = director.sample(n=min(N, len(director)), random_state=args.seed)
        if N and N < len(actor):
            actor = actor.sample(n=min(N, len(actor)), random_state=args.seed)
        print(f"Row sampling (non-sample mode): N={N}, seed={args.seed}")

    # -----------------------
    # Process director data
    # -----------------------
    print("Processing director data...")
    unique_people_director = director.drop_duplicates(
        subset=["person_odid", "movie_odid"]
    ).drop_duplicates(subset=["person_odid"])
    person_movie_pairs_director = list(
        zip(unique_people_director["person"], unique_people_director["display_name"])
    )

    print("Collecting director bios...")
    results = []
    for name, movie in tqdm(person_movie_pairs_director):
        result = fetch_bio_with_full_content(name, movie)
        results.append(
            {
                "name": name,
                "movie": movie,
                "status": result.get("status"),
                "title": result.get("title"),
                "url": result.get("url"),
                "summary": result.get("summary"),
                "early_life": result.get("early_life"),
                "full_content": result.get("content_snippet"),
            }
        )

    df_results_directors = pd.DataFrame(results)
    success_rate = (df_results_directors["status"] == "success").mean()
    print(f"Success rate (directors): {success_rate:.1%}")

    df_directors_success = df_results_directors[
        df_results_directors["status"] == "success"
    ]

    os.makedirs("wiki_bio_collected", exist_ok=True)
    df_directors_success.to_csv("wiki_bio_collected/directors_bio.csv", index=False)
    print("Saved director bios to wiki_bio_collected/directors_bio.csv")

    # -----------------------
    # Process actor data
    # -----------------------
    print("Processing actor data...")
    unique_people_actor = actor.drop_duplicates(
        subset=["person_odid", "movie_odid"]
    ).drop_duplicates(subset=["person_odid"])
    person_movie_pairs_actor = list(
        zip(unique_people_actor["person"], unique_people_actor["display_name"])
    )

    print("Collecting actor bios...")
    results_actor = []
    for name, movie in tqdm(person_movie_pairs_actor):
        result = fetch_bio_with_full_content(name, movie)
        results_actor.append(
            {
                "name": name,
                "movie": movie,
                "status": result.get("status"),
                "title": result.get("title"),
                "url": result.get("url"),
                "summary": result.get("summary"),
                "early_life": result.get("early_life"),
                "full_content": result.get("content_snippet"),
            }
        )

    df_results_actor = pd.DataFrame(results_actor)
    success_rate_actor = (df_results_actor["status"] == "success").mean()
    print(f"Success rate (actors): {success_rate_actor:.1%}")

    df_actor_success = df_results_actor[df_results_actor["status"] == "success"]
    df_actor_success.to_csv("wiki_bio_collected/actors_bio.csv", index=False)
    print("Saved actor bios to wiki_bio_collected/actors_bio.csv")

    # -----------------------
    # Summary output
    # -----------------------
    print("\n--- Summary ---")
    print(
        {
            "sampling": do_sample,
            "n": N if not do_sample else "file-based-sample",
            "success_rate_director": f"{success_rate:.1%}",
            "success_rate_actor": f"{success_rate_actor:.1%}",
            "actor_source": os.path.abspath(actor_path),
            "director_source": os.path.abspath(director_path),
        }
    )


if __name__ == "__main__":
    main()
