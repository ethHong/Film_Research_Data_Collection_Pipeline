#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge TMDB + Wikipedia (race tags) + FairFace predictions into one CSV.

- Inputs:
  * TMDB JSONs (default: TMDB_data_collected/actor_TMDB_data.json, director_TMDB_data.json)
  * Wikipedia race-tag CSVs (default: wiki_bio_collected/output/*.csv)
  * Original Excel (actor/director) to map person_odid (sample mode supported)
  * FairFace result CSV (default: fairface_result/race_prediction.csv)

- Output:
  * film_industry_complete_data.csv (in current working directory)

Notes:
  * SAMPLE mode: --sample or env SAMPLE=true
  * TMDB API key: --api_key or env TMDB_API_KEY
  * No Slack/webhook code included (removed)
"""

import os
import json
import argparse
import requests
import pandas as pd
import numpy as np


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Merge TMDB + Wikipedia race tags + FairFace predictions."
    )

    # TMDB JSON inputs
    p.add_argument(
        "--tmdb_dir",
        type=str,
        default="TMDB_data_collected",
        help="Directory containing TMDB JSON files.",
    )
    p.add_argument(
        "--tmdb_actor_json",
        type=str,
        default="actor_TMDB_data.json",
        help="Actor TMDB JSON filename (inside tmdb_dir).",
    )
    p.add_argument(
        "--tmdb_director_json",
        type=str,
        default="director_TMDB_data.json",
        help="Director TMDB JSON filename (inside tmdb_dir).",
    )

    # Wikipedia CSVs
    p.add_argument(
        "--wiki_dir",
        type=str,
        default="wiki_bio_collected/output",
        help="Directory containing wiki output CSVs.",
    )
    p.add_argument(
        "--wiki_actor_csv",
        type=str,
        default="actors_tag_from_wiki.csv",
        help="Wiki actors CSV filename (inside wiki_dir).",
    )
    p.add_argument(
        "--wiki_director_csv",
        type=str,
        default="director_tag_from_wiki.csv",
        help="Wiki directors CSV filename (inside wiki_dir).",
    )

    # Excel originals (for joining person_odid)
    p.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing original Excel files.",
    )

    # FairFace CSV
    p.add_argument(
        "--fairface_csv",
        type=str,
        default="fairface_result/race_prediction.csv",
        help="Path to FairFace prediction CSV.",
    )

    # TMDB API
    p.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="TMDB API key (or set TMDB_API_KEY env).",
    )

    # Output
    p.add_argument(
        "--out_csv",
        type=str,
        default="film_industry_complete_data.csv",
        help="Output CSV filename.",
    )

    # Mode: default = sample; use --full or SAMPLE=False to run full
    p.add_argument(
        "--full",
        action="store_true",
        help="Force full mode (override default sample mode).",
    )

    return p.parse_args()


# ---------------------------
# Helpers
# ---------------------------
# --- add to argparse ---


# --- replace sampling_enabled() ---
def sampling_enabled(args) -> bool:
    """Default to sample mode; disable if --full or env SAMPLE in falsey set."""
    # CLI has higher priority
    if args.full:
        return False
    # Env override (SAMPLE=False/0/no/n)
    env_val = os.getenv("SAMPLE", "").strip().lower()
    if env_val in {"0", "false", "f", "no", "n"}:
        return False
    # Default: sample mode
    return True


def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Fix string encoding from Latin-1 to UTF-8 for object columns."""
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(
            lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x
        )
    return df


def get_person_detail(person_id: str, api_key: str):
    """Fetch person detail from TMDB API."""
    base_url = "https://api.themoviedb.org/3/person/"
    url = f"{base_url}{person_id}"
    params = {"api_key": api_key, "language": "en-US"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        return r.json()
    else:
        # keep original behavior (print error) but don't raise
        print(f"Error {r.status_code}: {r.text}")
        return None


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    do_sample = sampling_enabled(args)

    # Resolve API key
    API_KEY = args.api_key or os.getenv("TMDB_API_KEY")
    if not API_KEY:
        raise RuntimeError(
            "TMDB API key not provided. Use --api_key or set TMDB_API_KEY env."
        )

    # TMDB JSON paths
    actor_json_path = os.path.join(args.tmdb_dir, args.tmdb_actor_json)
    director_json_path = os.path.join(args.tmdb_dir, args.tmdb_director_json)
    if not os.path.exists(actor_json_path):
        raise FileNotFoundError(f"Missing TMDB actor JSON: {actor_json_path}")
    if not os.path.exists(director_json_path):
        raise FileNotFoundError(f"Missing TMDB director JSON: {director_json_path}")

    # Load TMDB JSONs
    with open(actor_json_path, "r", encoding="utf-8") as f:
        TMDB_actor = json.load(f)
    with open(director_json_path, "r", encoding="utf-8") as f:
        TMDB_director = json.load(f)

    TMDB_data = {}
    TMDB_data.update(TMDB_actor)
    TMDB_data.update(TMDB_director)

    # Wiki CSVs
    wiki_actors_path = os.path.join(args.wiki_dir, args.wiki_actor_csv)
    wiki_directors_path = os.path.join(args.wiki_dir, args.wiki_director_csv)
    if not os.path.exists(wiki_actors_path):
        raise FileNotFoundError(f"Missing wiki actors CSV: {wiki_actors_path}")
    if not os.path.exists(wiki_directors_path):
        raise FileNotFoundError(f"Missing wiki directors CSV: {wiki_directors_path}")

    wiki_actors = pd.read_csv(wiki_actors_path)
    wiki_directors = pd.read_csv(wiki_directors_path)

    # Clean wiki strings (keep behavior)
    wiki_actors["name"] = wiki_actors["name"].str.replace("\n", " ").str.strip()
    wiki_actors["movie"] = wiki_actors["movie"].str.replace("\n", " ").str.strip()

    wiki_directors["name"] = wiki_directors["name"].str.replace("\n", " ").str.strip()
    wiki_directors["movie"] = wiki_directors["movie"].str.replace("\n", " ").str.strip()

    # Build id → name map (kept for parity, even if unused)
    id_name_map = {}
    for id, detail in TMDB_data.items():
        name = detail["person_name"]
        id_name_map[id] = name

    # Resolve Excel filenames based on SAMPLE mode
    data_dir = args.data_dir.rstrip("/")

    if do_sample:
        actor_xls = os.path.join(
            data_dir, "SAMPLED_Leading and Leading Ensemble Actor.xlsx"
        )
        director_xls = os.path.join(
            data_dir,
            "SAMPLED_Director-Producer-Exec_Producer-Screenwriter_sampled.xlsx",
        )
    else:
        actor_xls = os.path.join(data_dir, "Leading and Leading Ensemble Actor.xls")
        director_xls = os.path.join(
            data_dir, "Director-Producer-Exec_Producer-Screenwriter.xlsx"
        )

    if not os.path.exists(actor_xls):
        raise FileNotFoundError(f"Missing actor Excel: {actor_xls}")
    if not os.path.exists(director_xls):
        raise FileNotFoundError(f"Missing director Excel: {director_xls}")

    # Load original Excels and normalize
    original_actors = pd.read_excel(actor_xls)
    original_directors = pd.read_excel(director_xls)

    original_actors = fix_encoding(original_actors)
    original_actors["display_name"] = original_actors["display_name"].str.strip()

    original_directors = fix_encoding(original_directors)
    original_directors["display_name"] = original_directors["display_name"].str.strip()

    original_actors["person_odid"] = original_actors["person_odid"].astype("str")
    original_directors["person_odid"] = original_directors["person_odid"].astype("str")

    original_actors = original_actors[["display_name", "person", "person_odid"]]
    original_directors = original_directors[["display_name", "person", "person_odid"]]

    # Left join wiki ↔ originals to get person_odid
    merged_actor_wiki = wiki_actors.merge(
        original_actors,
        left_on=["name", "movie"],
        right_on=["person", "display_name"],
        how="left",
    ).drop(columns=["person", "display_name"])

    merged_director_wiki = wiki_directors.merge(
        original_directors,
        left_on=["name", "movie"],
        right_on=["person", "display_name"],
        how="left",
    ).drop(columns=["person", "display_name"])

    # TMDB dict → DataFrame
    TMDB_df = (
        pd.DataFrame.from_dict(TMDB_data, orient="index")
        .reset_index()
        .rename(columns={"index": "person_odid"})
    )

    TMDB_wiki_merged = TMDB_df.merge(
        merged_actor_wiki[["person_odid", "race_tag_final"]],
        on="person_odid",
        how="left",
    ).rename(columns={"race_tag_final": "wiki_race_tag_actors"})

    TMDB_wiki_merged = TMDB_wiki_merged.merge(
        merged_director_wiki[["person_odid", "race_tag_final"]],
        on="person_odid",
        how="left",
    ).rename(columns={"race_tag_final": "wiki_race_tag_directors"})

    # Gender & TMDB person id
    TMDB_wiki_merged["gender"] = (
        TMDB_wiki_merged["detail"]
        .apply(lambda x: x.get("gender") if isinstance(x, dict) else None)
        .fillna(0)
    )

    TMDB_wiki_merged["tmdb_id"] = TMDB_wiki_merged["detail"].apply(
        lambda x: str(x.get("id")) if isinstance(x, dict) else None
    )

    # Null stats (same prints)
    mask = TMDB_wiki_merged["image_url"].isna()
    count_nulls = mask.sum()
    percent = count_nulls / len(TMDB_wiki_merged) * 100
    print("Rows with no image:", count_nulls)
    print("Percentage of rows with image nulls:", percent)

    mask = (
        TMDB_wiki_merged["image_url"].isna()
        & TMDB_wiki_merged["wiki_race_tag_actors"].isna()
        & TMDB_wiki_merged["wiki_race_tag_directors"].isna()
    )
    count_nulls = mask.sum()
    percent = count_nulls / len(TMDB_wiki_merged) * 100
    print("Rows with all three null:", count_nulls)
    print("Percentage of nulls with wiki imputed data:", percent)

    # --- Collect DOB from TMDB API (keep original behavior, no Slack) ---
    from tqdm import tqdm

    n = len(TMDB_wiki_merged["tmdb_id"].values)
    step = max(1, n // 20)

    DOB = []
    for i, ids in enumerate(tqdm(TMDB_wiki_merged["tmdb_id"].values)):
        if pd.isna(ids) or ids is None:
            birthday = None
        else:
            detail = get_person_detail(ids, API_KEY)
            if detail and "birthday" in detail:
                birthday = detail["birthday"]
            else:
                birthday = None
        DOB.append(birthday)

        # Preserve the occasional progress print (without Slack)
        if i % step == 0:
            try:
                name = TMDB_wiki_merged.loc[TMDB_wiki_merged["tmdb_id"] == ids][
                    "person_name"
                ].values[0]
            except Exception:
                name = None
            print(
                f"Progress {int(i/n*100)}% ({i}/{n}) | Actor: {name}, DOB: {birthday}"
            )

    TMDB_wiki_merged.to_csv("TMDB_wiki_merged_with_DOB.csv", index=False)
    print("Saved intermediate file: TMDB_wiki_merged_with_DOB.csv")

    TMDB_wiki_merged["DOB"] = DOB

    # --- Merge FairFace ---
    fairface = pd.read_csv(args.fairface_csv)
    fairface = fairface[fairface["face_name_align"].str.contains("face0")]
    fairface["person_odid"] = fairface["face_name_align"].apply(
        lambda x: x.split("/")[2].split("_")[0]
    )

    fairface_merged = fairface.drop_duplicates("person_odid")
    fairface_merged = fairface_merged[
        [
            "person_odid",
            "race",
            "race4",
            "gender",
            "age",
            "race_scores_fair",
            "race_scores_fair_4",
            "age_scores_fair",
            "gender_scores_fair",
        ]
    ].rename(
        columns={
            "race": "race_predicted",
            "race4": "race4_predicted",
            "gender": "gender_predicted",
            "age": "age_predicted",
        }
    )

    TMDB_wiki_merged.person_odid = TMDB_wiki_merged.person_odid.astype("str")
    fairface_merged.person_odid = fairface_merged.person_odid.astype("str")

    merged = TMDB_wiki_merged.merge(fairface_merged, on="person_odid", how="left")

    # Flags for membership
    actors_odid = original_actors.person_odid.unique()
    directors_odid = original_directors.person_odid.unique()

    merged["in_actorlist"] = merged["person_odid"].isin(actors_odid)
    merged["in_directorlist"] = merged["person_odid"].isin(directors_odid)

    # Drop duplicates by person_odid
    merged = merged.drop_duplicates(subset=["person_odid"], keep="first")

    # Final output
    merged.to_csv(args.out_csv, index=False)
    print(f"Saved final output: {args.out_csv}")


if __name__ == "__main__":
    main()
