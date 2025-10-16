#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map race tags to processed Wikipedia bios using keyword and country mapping.

- Input: wiki_bio_collected/ directory (expects 'processed/' subdir and mapping Excel files)
- Output: actors_tag_from_wiki.csv, director_tag_from_wiki.csv in wiki_bio_collected/
"""

import pandas as pd
import os
import ast
import argparse


# ---------------------------
# CLI argument configuration
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Map race tags from wiki bio processed data."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="wiki_bio_collected",
        help="Input directory containing processed CSVs and mapping Excel files.",
    )
    parser.add_argument(
        "--keyword_map",
        type=str,
        default="race_keyword_map.xlsx",
        help="Excel file mapping keywords → race tags (inside in_dir).",
    )
    parser.add_argument(
        "--country_map",
        type=str,
        default="race_country_mapped.xlsx",
        help="Excel file mapping countries → race tags (inside in_dir).",
    )
    return parser.parse_args()


# ---------------------------
# Helper functions
# ---------------------------
def safe_literal_eval(x):
    """Convert stringified list to Python list safely."""
    try:
        return ast.literal_eval(x) if pd.notnull(x) else None
    except Exception:
        return None


def get_race_tag_keywords(keyword_list, mapping):
    """Map a list of keywords to race tags."""
    races = []
    if not isinstance(keyword_list, list):
        return []
    for kw in keyword_list:
        if kw in mapping:
            mapped = [tag.strip() for tag in mapping[kw][0].split(",")]
            for tag in mapped:
                if tag not in races and tag != "Irrelevant":
                    races.append(tag)
    return races


def get_race_tag_country(word, mapping):
    """Map a single country keyword to race tags."""
    if not isinstance(word, str):
        return []
    races = []
    if word in mapping:
        mapped = [tag.strip() for tag in mapping[word][0].split(",")]
        for tag in mapped:
            if tag not in races and tag != "Irrelevant":
                races.append(tag)
    return races


def compute_final_tag(tags):
    """Return a single race label ('None', 'Mixed', or one tag)."""
    if not tags:
        return "None"
    return "Mixed" if len(tags) > 1 else tags[0]


# ---------------------------
# Main processing function
# ---------------------------
def process_race_tags(df, keywords_mapping, country_mapping):
    """Apply race tag extraction to a single DataFrame."""
    # Step 1: extract from keyword-based race hints
    df["merged_race_keywords"] = df["merged_race_keywords"].apply(safe_literal_eval)
    df["race_tags"] = df["merged_race_keywords"].apply(
        lambda x: get_race_tag_keywords(x, keywords_mapping) if x is not None else None
    )

    # Step 2: if keyword tags missing, use country-based mapping
    df["race_tags"] = df.apply(
        lambda row: (
            row["race_tags"]
            if row["race_tags"] is not None and len(row["race_tags"]) > 0
            else get_race_tag_country(row["country_hint"], country_mapping)
        ),
        axis=1,
    )

    # Step 3: finalize
    df["race_tag_final"] = df["race_tags"].apply(compute_final_tag)
    return df


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    in_dir = args.in_dir.rstrip("/")
    processed_dir = os.path.join(in_dir, "processed")

    # File paths
    actor_csv = os.path.join(
        processed_dir, "actors_bio_hint_phrases_processed_final_merged.csv"
    )
    director_csv = os.path.join(
        processed_dir, "directors_bio_hint_phrases_processed_final_merged.csv"
    )
    keyword_map_path = os.path.join(in_dir, args.keyword_map)
    country_map_path = os.path.join(in_dir, args.country_map)

    # Validate files
    for f in [actor_csv, director_csv, keyword_map_path, country_map_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    print("Loading input files...")
    actors = pd.read_csv(actor_csv)
    directors = pd.read_csv(director_csv)

    # Load mapping Excel files
    print("Loading mapping dictionaries...")
    keywords_map = pd.read_excel(keyword_map_path)
    keywords_map = keywords_map.loc[:, ~keywords_map.columns.duplicated()]
    country_map = pd.read_excel(country_map_path)
    country_map = country_map.loc[:, ~country_map.columns.duplicated()]

    keywords_mapping = (
        keywords_map[["Keyword", "Race Tag"]].set_index("Keyword").T.to_dict("list")
    )
    # Manual correction example
    keywords_mapping["Italian"] = ["White"]

    country_mapping = (
        country_map[["keyword", "race_tag"]].set_index("keyword").T.to_dict("list")
    )

    # Process both dataframes
    print("Processing actors...")
    actors = process_race_tags(actors, keywords_mapping, country_mapping)

    print("Processing directors...")
    directors = process_race_tags(directors, keywords_mapping, country_mapping)

    # Output results
    output_dir = os.path.join(in_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    actors_out = os.path.join(output_dir, "actors_tag_from_wiki.csv")
    directors_out = os.path.join(output_dir, "director_tag_from_wiki.csv")

    actors.to_csv(actors_out, index=False)
    directors.to_csv(directors_out, index=False)

    # Summary
    actors_total = actors["race_tag_final"].notnull().sum()
    directors_total = directors["race_tag_final"].notnull().sum()

    print("\n--- Summary ---")
    print(
        {
            "actors_rows": len(actors),
            "directors_rows": len(directors),
            "actors_tagged": actors_total,
            "directors_tagged": directors_total,
            "output_dir": in_dir,
        }
    )


if __name__ == "__main__":
    main()
