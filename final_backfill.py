#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final backfill and aggregation pipeline.

- Input:
  * film_industry_complete_data.csv (from merge step)
  * wiki_bio_collected/actors_bio.csv, wiki_bio_collected/directors_bio.csv
  * Original Excel files in ./data (sample or full based on mode)

- Output (to Final_Output/, prefix with SAMPLE_ if sample mode):
  * <prefix>final_full_data.csv
  * <prefix>movie_ppl_aggregate.csv
  * <prefix>movie_ppl_mainrole_DirectorProducer_aggregate.csv
  * <prefix>movie_ppl_mainrole_DirectorScreenwriter_aggregate.csv

Mode:
  * Default is SAMPLE mode. Use --full or env SAMPLE=False to run full dataset.
"""

import os
import re
import ast
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------
# CLI & mode handling
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Final backfill and aggregation.")
    p.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing original Excel files.",
    )
    p.add_argument(
        "--in_csv",
        type=str,
        default="film_industry_complete_data.csv",
        help="Input merged CSV (from merge_wiki_tmdb_fairface.py).",
    )
    p.add_argument(
        "--wiki_dir",
        type=str,
        default="wiki_bio_collected",
        help="Directory containing actors_bio.csv and directors_bio.csv.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="Final_Output",
        help="Output directory for final CSVs.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Force full mode (override default sample mode).",
    )
    return p.parse_args()


def sampling_enabled(args) -> bool:
    """Default to sample mode; disable if --full or env SAMPLE in falsey set."""
    if args.full:
        return False
    env_val = os.getenv("SAMPLE", "").strip().lower()
    if env_val in {"0", "false", "f", "no", "n"}:
        return False
    return True


# ---------------------------
# Helpers
# ---------------------------
def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Fix encoding issues for object columns (latin1 -> utf-8)."""
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(
            lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x
        )
    return df


def clean_movie_list(raw_value):
    """Normalize movie list field that may contain weird serialized formats."""
    cleaned = []
    text = str(raw_value).strip()
    try:
        movies = ast.literal_eval(text)
        if isinstance(movies, list) and all(
            isinstance(x, str) and len(x) == 1 for x in movies
        ):
            joined = "".join(movies)
            inner = re.sub(r"^\[|\]$", "", joined)
            candidates = [c.strip(" '\"") for c in inner.split(",") if c.strip(" '\"")]
            cleaned.extend(candidates)
        elif isinstance(movies, list):
            cleaned.extend([str(m).strip(" '\"") for m in movies])
        else:
            cleaned.append(str(movies).strip(" '\""))
    except Exception:
        inner = re.sub(r"^\[|\]$", "", text)
        candidates = [c.strip(" '\"") for c in inner.split(",") if c.strip(" '\"")]
        cleaned.extend(candidates)
    return cleaned


# --- DOB extraction utilities (kept behavior) ---
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _norm(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _safe_dt(y: int, m: int, d: int) -> str | None:
    try:
        return _norm(datetime(y, m, d))
    except ValueError:
        return None


def _parse_month_name(name: str) -> int | None:
    return _MONTHS.get(name.lower())


_RE_ISO = re.compile(r"\b(18|19|20)\d{2}-\d{2}-\d{2}\b")
_RE_MONTH_DD_YYYY = re.compile(
    r"(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2}),?\s+(?P<year>(?:18|19|20)\d{2})",
    flags=re.IGNORECASE,
)
_RE_DD_MONTH_YYYY = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9}),?\s+(?P<year>(?:18|19|20)\d{2})",
    flags=re.IGNORECASE,
)
_RE_YEAR = re.compile(r"\b(18|19|20)\d{2}\b")
_POS_HINT = re.compile(r"\b(born|birth|né|née|b\.)\b", re.IGNORECASE)
_NEG_HINT = re.compile(
    r"\b(publish(?:ed|ing)?|released?|novel|movie|book|film|album|created|founded|since|in\s+the\s+year)\b",
    re.IGNORECASE,
)


def _try_parse_date_piece(s: str) -> str | None:
    m = _RE_ISO.search(s)
    if m:
        y, mo, d = m.group(0).split("-")
        return _safe_dt(int(y), int(mo), int(d))
    m = _RE_MONTH_DD_YYYY.search(s)
    if m:
        mo = _parse_month_name(m.group("month"))
        if mo:
            return _safe_dt(int(m.group("year")), mo, int(m.group("day")))
    m = _RE_DD_MONTH_YYYY.search(s)
    if m:
        mo = _parse_month_name(m.group("month"))
        if mo:
            return _safe_dt(int(m.group("year")), mo, int(m.group("day")))
    return None


def extract_dob(name: str, text: str) -> str | None:
    """DOB extraction (kept original behavior with improvements)."""
    if not text or not name:
        return None

    escaped_name = re.escape(name)

    # 0-a) Parentheses after the name where 'born' appears anywhere inside
    born_anywhere_pat = re.compile(
        rf"{escaped_name}\s*\(\s*([^)]+born[^)]*)\)", re.IGNORECASE
    )
    m0a = born_anywhere_pat.search(text)
    if m0a:
        inside_all = m0a.group(1)
        if not _NEG_HINT.search(inside_all):
            mb = re.search(r"born\s+([^)]+)", inside_all, flags=re.IGNORECASE)
            if mb:
                born_tail = mb.group(1)
                d = _try_parse_date_piece(born_tail)
                if d:
                    return d
                y = _RE_YEAR.search(born_tail)
                if y:
                    yyyy = int(y.group(0))
                    this_year = datetime.now().year
                    if 1850 <= yyyy <= this_year:
                        return _safe_dt(yyyy, 1, 1)

    # 0) Parentheses right after the name
    paren_pat = re.compile(
        rf"{escaped_name}\s*\(\s*(?:born\s+)?(?P<inside>[^){{0,140}}]+)\)",
        re.IGNORECASE,
    )
    m = paren_pat.search(text)
    if m:
        inside = m.group("inside")
        if not _NEG_HINT.search(inside):
            d = _try_parse_date_piece(inside)
            if d:
                return d
            y = _RE_YEAR.search(inside)
            if y:
                yyyy = int(y.group(0))
                this_year = datetime.now().year
                if 1850 <= yyyy <= this_year:
                    return _safe_dt(yyyy, 1, 1)

    # 1) Dates near positive keywords (right-side window)
    WINDOW = 120
    for hit in _POS_HINT.finditer(text):
        right = text[hit.end() : min(len(text), hit.end() + WINDOW)]
        stop = re.search(r"[.;:?!\)\]]", right)
        if stop:
            right = right[: stop.start()]
        if _NEG_HINT.search(right):
            continue
        d = _try_parse_date_piece(right)
        if d:
            return d
        y = _RE_YEAR.search(right)
        if y:
            yyyy = int(y.group(0))
            this_year = datetime.now().year
            if 1850 <= yyyy <= this_year:
                return _safe_dt(yyyy, 1, 1)

    # 2) Fallback: first block (intro)
    first_block = text.split("\n\n", 1)[0]
    if name in first_block:
        m = re.search(r"\(([^)]{1,140})\)", first_block)
        if m and not _NEG_HINT.search(m.group(1)):
            d = _try_parse_date_piece(m.group(1))
            if d:
                return d

    return None


_RE_GENDER = re.compile(
    r"\b(actress|female|she|her)\b|\b(actor|male|he|him|his)\b", re.IGNORECASE
)


def extract_gender(text: str) -> str | None:
    """Very simple gender cue extractor."""
    if not text:
        return None
    m = _RE_GENDER.search(text)
    if not m:
        return None
    token = m.group(0).lower()
    if token in ["actress", "female", "she", "her"]:
        return "Female"
    if token in ["actor", "male", "he", "him", "his"]:
        return "Male"
    return None


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    do_sample = sampling_enabled(args)
    prefix = "SAMPLE_" if do_sample else ""

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"[Init] Mode: {'SAMPLE' if do_sample else 'FULL'} | Output dir: {args.out_dir}"
    )

    # 1) Load merged input
    print("[1/7] Loading merged input CSV...")
    df = pd.read_csv(args.in_csv)

    # Split convenience sets
    haveRace = df[
        (
            (~df["wiki_race_tag_actors"].isnull())
            | (~df["wiki_race_tag_directors"].isnull())
            | (~df["race_predicted"].isnull())
        )
    ]
    haveRace_noGender = df[
        (
            (~df["wiki_race_tag_actors"].isnull())
            | (~df["wiki_race_tag_directors"].isnull())
            | (~df["race_predicted"].isnull())
        )
        & ((df["gender"] == 0) & (df["gender_predicted"].isnull()))
    ]
    haveRace_noDOB = df[
        (
            (~df["wiki_race_tag_actors"].isnull())
            | (~df["wiki_race_tag_directors"].isnull())
            | (~df["race_predicted"].isnull())
        )
        & (df["DOB"].isnull())
    ]

    # 2) Load wiki bios (actors/directors)
    print("[2/7] Loading wiki bios CSVs...")
    wiki_actors = pd.read_csv(os.path.join(args.wiki_dir, "actors_bio.csv"))
    wiki_directors = pd.read_csv(os.path.join(args.wiki_dir, "directors_bio.csv"))

    # Build name → {odid: [movies]} map from merged df
    print("[3/7] Building name→odid→movies map...")
    name_odid_movies = {}
    for odid, name, movies in zip(df["person_odid"], df["person_name"], df["movies"]):
        if name not in name_odid_movies:
            name_odid_movies[name] = {}
        name_odid_movies[name][odid] = clean_movie_list(movies)

    def get_right_odid(name, movie):
        temp = name_odid_movies.get(name, {})
        if not temp:
            return None
        if len(temp) == 1:
            for id_, movie_list in temp.items():
                return id_
        else:
            for id_, movie_list in temp.items():
                if movie in movie_list:
                    return id_
            return None

    actor_ids = [
        get_right_odid(n, m) for n, m in zip(wiki_actors["name"], wiki_actors["movie"])
    ]
    director_ids = [
        get_right_odid(n, m)
        for n, m in zip(wiki_directors["name"], wiki_directors["movie"])
    ]

    wiki_actors["person_odid"] = actor_ids
    wiki_directors["person_odid"] = director_ids

    wiki_concat = pd.concat([wiki_actors, wiki_directors], ignore_index=True)
    wiki_concat = wiki_concat.drop_duplicates(subset=["person_odid"])

    # 3) Backfill DOB from wiki text
    print("[4/7] Backfilling DOB from wiki text...")
    haveRace_noDOB = haveRace_noDOB.merge(
        wiki_concat[["url", "summary", "early_life", "full_content", "person_odid"]],
        on="person_odid",
        how="left",
    )
    haveRace_noDOB_backfill = haveRace_noDOB[~haveRace_noDOB["url"].isnull()].copy()

    haveRace_noDOB_backfill["DOB_wiki_backfill"] = [
        extract_dob(name, summary)
        for name, summary in zip(
            haveRace_noDOB_backfill["person_name"], haveRace_noDOB_backfill["summary"]
        )
    ]
    haveRace_noDOB_backfill["DOB_wiki_backfill_early_life"] = [
        extract_dob(name, early_life) if pd.notnull(early_life) else None
        for name, early_life in zip(
            haveRace_noDOB_backfill["person_name"],
            haveRace_noDOB_backfill["early_life"],
        )
    ]
    haveRace_noDOB_backfill = haveRace_noDOB_backfill[
        (~haveRace_noDOB_backfill["DOB_wiki_backfill"].isnull())
        | (~haveRace_noDOB_backfill["DOB_wiki_backfill_early_life"].isnull())
    ]
    haveRace_noDOB_backfill["DOB_wiki_backfill_final"] = [
        dob if pd.notnull(dob) else dob_early_life
        for dob, dob_early_life in zip(
            haveRace_noDOB_backfill["DOB_wiki_backfill"],
            haveRace_noDOB_backfill["DOB_wiki_backfill_early_life"],
        )
    ]

    df = df.merge(
        haveRace_noDOB_backfill[["person_odid", "DOB_wiki_backfill_final"]],
        on="person_odid",
        how="left",
    )
    added_DOB = df[
        (df["DOB"].isnull()) & (~df["DOB_wiki_backfill_final"].isnull())
    ].shape[0]
    print(f"[DOB] Added {added_DOB} DOBs")

    # 4) Backfill Gender from wiki text
    print("[5/7] Backfilling Gender from wiki text...")
    haveRace_noGender = haveRace_noGender.merge(
        wiki_concat[["url", "summary", "early_life", "full_content", "person_odid"]],
        on="person_odid",
        how="left",
    )
    haveRace_noGender_backfill = haveRace_noGender[
        ~haveRace_noGender["url"].isnull()
    ].copy()

    haveRace_noGender_backfill["Gender_wiki_backfill"] = [
        extract_gender(summary) for summary in haveRace_noGender_backfill["summary"]
    ]
    haveRace_noGender_backfill["Gender_wiki_backfill_early_life"] = [
        extract_gender(early_life) if pd.notnull(early_life) else None
        for early_life in haveRace_noGender_backfill["early_life"]
    ]
    haveRace_noGender_backfill = haveRace_noGender_backfill[
        (~haveRace_noGender_backfill["Gender_wiki_backfill"].isnull())
        | (~haveRace_noGender_backfill["Gender_wiki_backfill_early_life"].isnull())
    ]
    haveRace_noGender_backfill["Gender_wiki_backfill_final"] = [
        g if pd.notnull(g) else g2
        for g, g2 in zip(
            haveRace_noGender_backfill["Gender_wiki_backfill"],
            haveRace_noGender_backfill["Gender_wiki_backfill_early_life"],
        )
    ]

    df = df.merge(
        haveRace_noGender_backfill[["person_odid", "Gender_wiki_backfill_final"]],
        on="person_odid",
        how="left",
    )
    added_gender = df[
        (df["gender"] == 0) & (~df["Gender_wiki_backfill_final"].isnull())
    ].shape[0]
    print(f"[Gender] Added {added_gender} genders")

    # 5) Source consolidation
    print("[6/7] Consolidating sources and computing movie-level aggregates...")
    df["DOB_source"] = [
        (
            "TMDB"
            if pd.notnull(dob)
            else ("Wiki_document" if pd.notnull(wiki_dob) else "Unknown")
        )
        for dob, wiki_dob in zip(df["DOB"], df["DOB_wiki_backfill_final"])
    ]
    df["DOB"] = [
        dob if pd.notnull(dob) else dob_backfill
        for dob, dob_backfill in zip(df["DOB"], df["DOB_wiki_backfill_final"])
    ]

    df["Gender_source"] = [
        (
            "TMDB"
            if gender != 0
            else (
                "Wiki_document"
                if pd.notnull(wiki_gender)
                else (
                    "TMDB Image Prediction Model"
                    if pd.notnull(gender_pred)
                    else "Unknown"
                )
            )
        )
        for gender, wiki_gender, gender_pred in zip(
            df["gender"], df["Gender_wiki_backfill_final"], df["gender_predicted"]
        )
    ]
    df["Gender"] = [
        "Female" if g == 1 else ("Male" if g == 2 else None) for g in df["gender"]
    ]
    df["Gender"] = [
        (
            g
            if g in ("Female", "Male")
            else (wg if pd.notnull(wg) else (gp if pd.notnull(gp) else None))
        )
        for g, wg, gp in zip(
            df["Gender"], df["Gender_wiki_backfill_final"], df["gender_predicted"]
        )
    ]

    df["wiki_race_tag"] = [
        actors if pd.notnull(actors) else directors
        for actors, directors in zip(
            df["wiki_race_tag_actors"], df["wiki_race_tag_directors"]
        )
    ]

    # Save final_full_data
    final_full_path = os.path.join(args.out_dir, f"{prefix}final_full_data.csv")
    df_final = df[
        [
            "person_odid",
            "person_name",
            "DOB",
            "DOB_source",
            "Gender",
            "Gender_source",
            "wiki_race_tag",
            "race_predicted",
            "race4_predicted",
            "race_scores_fair",
            "race_scores_fair_4",
            "in_actorlist",
            "in_directorlist",
        ]
    ].copy()
    df_final.to_csv(final_full_path, index=False)
    print(f"[Save] People-level file: {final_full_path}")

    # 6) Aggregate to movie-level
    # Load original excels (sample vs full)
    if do_sample:
        actor_xls = os.path.join(
            args.data_dir, "SAMPLED_Leading and Leading Ensemble Actor.xlsx"
        )
        director_xls = os.path.join(
            args.data_dir,
            "SAMPLED_Director-Producer-Exec_Producer-Screenwriter_sampled.xlsx",
        )
    else:
        actor_xls = os.path.join(
            args.data_dir, "Leading and Leading Ensemble Actor.xls"
        )
        director_xls = os.path.join(
            args.data_dir, "Director-Producer-Exec_Producer-Screenwriter.xlsx"
        )

    print("[6.1] Loading original Excel files for aggregation...")
    original_actors = pd.read_excel(actor_xls)
    original_directors = pd.read_excel(director_xls)
    original_actors = fix_encoding(original_actors)
    original_directors = fix_encoding(original_directors)
    original_actors["display_name"] = original_actors["display_name"].str.strip()
    original_directors["display_name"] = original_directors["display_name"].str.strip()

    original_concat = pd.concat(
        [original_actors, original_directors], ignore_index=True
    )

    total_aggregate = original_concat.merge(
        df_final, on="person_odid", how="left"
    ).sort_values("movie_odid")
    total_aggregate["Race_present"] = (
        total_aggregate[["wiki_race_tag", "race_predicted"]].notnull().any(axis=1)
    )
    total_aggregate["DOB_present"] = total_aggregate["DOB"].notnull()
    total_aggregate["Gender_present"] = total_aggregate["Gender"].notnull()

    coverage = (
        total_aggregate.groupby(["movie_odid", "display_name"])
        .agg(
            total_people=("person_odid", "count"),
            dob_coverage=("DOB_present", "mean"),
            gender_coverage=("Gender_present", "mean"),
            race_coverage=("Race_present", "mean"),
        )
        .reset_index()
    )
    total_aggregate = total_aggregate.merge(
        coverage, on=["movie_odid", "display_name"], how="left"
    )

    # Race mapping & confidence handling
    def map_race(input_race):
        racemap = {
            "White": "White",
            "Black": "Black",
            "Asian": "Asian",
            "Native American": "Native American / Pacific Islander",
            "Native Hawaiian or Other Pacific Islander": "Native American / Pacific Islander",
            "Mixed": "Mixed",
            "Other": "Other",
            "Latino / Hispanic": "Latino_Hispanic",
            "Latino_Hispanic": "Latino_Hispanic",
            "Middle Eastern": "White",
            "Indian": "Asian",
            "East Asian": "Asian",
            "Southeast Asian": "Asian",
        }
        return racemap.get(input_race, input_race)

    total_aggregate["wiki_race_tag"] = total_aggregate["wiki_race_tag"].apply(map_race)
    total_aggregate["race_predicted"] = total_aggregate["race_predicted"].apply(
        map_race
    )

    def _parse_scores(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            try:
                return [float(x) for x in val]
            except Exception:
                return np.nan
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return np.nan
        # Try Python/JSON literal
        try:
            lit = ast.literal_eval(s)
            if isinstance(lit, (list, tuple)):
                return [float(x) for x in lit]
        except Exception:
            pass
        # Fallback: replace commas with spaces and regex-extract numbers
        cleaned = s.strip("[]").replace(",", " ")
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        return [float(x) for x in nums] if nums else np.nan

    total_aggregate["race_scores_fair"] = total_aggregate["race_scores_fair"].apply(
        _parse_scores
    )
    total_aggregate["race_scores_max"] = total_aggregate["race_scores_fair"].apply(
        lambda x: max(x) if isinstance(x, list) and len(x) > 0 else np.nan
    )

    # Resolve final race based on wiki vs model and score threshold
    total_aggregate["race_final"] = total_aggregate.apply(
        lambda r: (
            r["wiki_race_tag"]
            if pd.notnull(r["wiki_race_tag"]) and pd.isnull(r["race_predicted"])
            else (
                r["race_predicted"]
                if pd.notnull(r["race_predicted"]) and pd.isnull(r["wiki_race_tag"])
                else (
                    r["wiki_race_tag"]
                    if pd.notnull(r["wiki_race_tag"])
                    and r.get("race_scores_max", np.nan) < 0.7
                    else r["race_predicted"]
                )
            )
        ),
        axis=1,
    )

    total_aggregate["race_source"] = total_aggregate.apply(
        lambda r: (
            "Wikipedia"
            if (
                (pd.notnull(r["wiki_race_tag"]) and pd.isnull(r["race_predicted"]))
                or (
                    pd.notnull(r["wiki_race_tag"])
                    and pd.notnull(r["race_predicted"])
                    and r.get("race_scores_max", np.nan) < 0.7
                )
            )
            else (
                "TMDB_Image_FairfaceModel"
                if pd.notnull(r["race_predicted"])
                else np.nan
            )
        ),
        axis=1,
    )

    # Latino/Hispanic mapping into White/Black based on first two probs
    mask = total_aggregate["race_final"].eq("Latino_Hispanic")
    total_aggregate.loc[mask, "race_final"] = total_aggregate.loc[
        mask, "race_scores_fair"
    ].apply(
        lambda x: (
            "Latino_Hispanic"
            if not (
                isinstance(x, (list, np.ndarray))
                and len(x) > 1
                and pd.notnull(x[0])
                and pd.notnull(x[1])
            )
            else (
                "White"
                if float(x[0]) > float(x[1])
                else ("Black" if float(x[1]) > float(x[0]) else "Latino_Hispanic")
            )
        )
    )

    cols = [
        "movie_odid",
        "display_name",
        "billing",
        "person_odid",
        "person",
        "character",
        "type",
        "role",
        "person_name",
        "DOB",
        "DOB_source",
        "Gender",
        "Gender_source",
        "race_final",
        "race_source",
        "race_scores_fair",
        "in_actorlist",
        "in_directorlist",
        "Race_present",
        "DOB_present",
        "Gender_present",
        "total_people",
        "dob_coverage",
        "gender_coverage",
        "race_coverage",
    ]

    # Save aggregates
    agg_path = os.path.join(args.out_dir, f"{prefix}movie_ppl_aggregate.csv")
    total_aggregate[cols].to_csv(agg_path, index=False)
    print(f"[Save] Movie-people aggregate: {agg_path}")

    # Director/Producer main roles
    total_aggregate_mainroles = total_aggregate.drop(
        ["total_people", "dob_coverage", "gender_coverage", "race_coverage"], axis=1
    )
    total_aggregate_mainroles = total_aggregate_mainroles[
        ((total_aggregate_mainroles["in_actorlist"]))
        | (
            (total_aggregate_mainroles["in_directorlist"])
            & (total_aggregate_mainroles["role"].isin(["Director", "Producer"]))
        )
    ]
    coverage_mainroles = (
        total_aggregate_mainroles.groupby(["movie_odid", "display_name"])
        .agg(
            total_people=("person_odid", "count"),
            dob_coverage=("DOB_present", "mean"),
            gender_coverage=("Gender_present", "mean"),
            race_coverage=("Race_present", "mean"),
        )
        .reset_index()
    )
    total_aggregate_mainroles = total_aggregate_mainroles.merge(
        coverage_mainroles, on=["movie_odid", "display_name"], how="left"
    )
    dp_path = os.path.join(
        args.out_dir, f"{prefix}movie_ppl_mainrole_DirectorProducer_aggregate.csv"
    )
    total_aggregate_mainroles[cols].to_csv(dp_path, index=False)
    print(f"[Save] Main roles (Director/Producer): {dp_path}")

    # Director/Screenwriter main roles
    total_aggregate_mainroles2 = total_aggregate.drop(
        ["total_people", "dob_coverage", "gender_coverage", "race_coverage"], axis=1
    )
    total_aggregate_mainroles2 = total_aggregate_mainroles2[
        ((total_aggregate_mainroles2["in_actorlist"]))
        | (
            (total_aggregate_mainroles2["in_directorlist"])
            & (total_aggregate_mainroles2["role"].isin(["Director", "Screenwriter"]))
        )
    ]
    coverage_mainroles2 = (
        total_aggregate_mainroles2.groupby(["movie_odid", "display_name"])
        .agg(
            total_people=("person_odid", "count"),
            dob_coverage=("DOB_present", "mean"),
            gender_coverage=("Gender_present", "mean"),
            race_coverage=("Race_present", "mean"),
        )
        .reset_index()
    )
    total_aggregate_mainroles2 = total_aggregate_mainroles2.merge(
        coverage_mainroles2, on=["movie_odid", "display_name"], how="left"
    )
    ds_path = os.path.join(
        args.out_dir, f"{prefix}movie_ppl_mainrole_DirectorScreenwriter_aggregate.csv"
    )
    total_aggregate_mainroles2[cols].to_csv(ds_path, index=False)
    print(f"[Save] Main roles (Director/Screenwriter): {ds_path}")

    print("[Done] Final backfill and aggregation complete.")


if __name__ == "__main__":
    main()
