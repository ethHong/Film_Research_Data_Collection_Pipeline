#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download TMDB profile images for actors and directors from pre-collected JSONs.

- Input (default: TMDB_data_collected/):
    actor_TMDB_data.json
    director_TMDB_data.json

- Output (default under TMDB_data_collected/):
    img_actors/<person_odid>.jpg
    img_directors/<person_odid>.jpg

- Behavior:
    * Creates output directories if missing
    * Skips entries with no `detail` or no `profile_path`
    * Optionally skips download if file already exists (--skip_existing)
    * Writes updated JSONs with `image_url` field (unless --no_update_json)
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Optional

import requests
from tqdm import tqdm


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TMDB profile images from collected JSONs."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="TMDB_data_collected",
        help="Directory containing actor_TMDB_data.json and director_TMDB_data.json.",
    )
    parser.add_argument(
        "--actors_json",
        type=str,
        default="actor_TMDB_data.json",
        help="Filename of the actors JSON (inside --in_dir).",
    )
    parser.add_argument(
        "--directors_json",
        type=str,
        default="director_TMDB_data.json",
        help="Filename of the directors JSON (inside --in_dir).",
    )
    parser.add_argument(
        "--out_actors_dir",
        type=str,
        default="img_actors",
        help="Subdirectory (under --in_dir) to save actor images.",
    )
    parser.add_argument(
        "--out_directors_dir",
        type=str,
        default="img_directors",
        help="Subdirectory (under --in_dir) to save director images.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="w500",
        help="TMDB image size (e.g., w185, w342, w500, original).",
    )
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout seconds.")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Sleep seconds between downloads."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip download if image file already exists.",
    )
    parser.add_argument(
        "--no_update_json",
        action="store_true",
        help="Do not write back image_url to the JSON files.",
    )
    return parser.parse_args()


# ---------------------------
# Helpers
# ---------------------------
def build_image_url(profile_path: str, size: str = "w500") -> str:
    """Build full TMDB image URL from profile_path and size."""
    base = "https://image.tmdb.org/t/p"
    size = size.lstrip("/")  # sanitize
    path = profile_path if profile_path.startswith("/") else f"/{profile_path}"
    return f"{base}/{size}{path}"


def download_image(url: str, out_path: str, timeout: int = 10) -> bool:
    """Download image from URL to out_path. Returns True on success."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def ensure_dir(path: str):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file into dict."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: str, data: Dict[str, Any]):
    """Save dict to JSON file."""
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)


def process_group(
    data: Dict[str, Any],
    img_dir: str,
    size: str,
    timeout: int,
    sleep_s: float,
    skip_existing: bool,
) -> int:
    """Process a single JSON dict: download images and set image_url. Returns count downloaded."""
    ensure_dir(img_dir)
    downloaded = 0

    # tqdm over keys to show deterministic progress
    for key in tqdm(list(data.keys())):
        entry = data.get(key, {})
        detail = entry.get("detail") if isinstance(entry, dict) else None

        # No detail or no profile path -> set None and continue
        if not detail or not isinstance(detail, dict):
            entry["image_url"] = None
            data[key] = entry
            continue

        profile_path = detail.get("profile_path")
        if not profile_path:
            entry["image_url"] = None
            data[key] = entry
            continue

        # Compose URL and output filename
        url = build_image_url(profile_path, size=size)
        entry["image_url"] = url
        data[key] = entry

        out_file = os.path.join(img_dir, f"{key}.jpg")
        if skip_existing and os.path.exists(out_file):
            # Skip existing file
            continue

        ok = download_image(url, out_file, timeout=timeout)
        if ok:
            downloaded += 1

        if sleep_s > 0:
            time.sleep(sleep_s)

    return downloaded


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    in_dir = args.in_dir.rstrip("/")
    actors_json_path = os.path.join(in_dir, args.actors_json)
    directors_json_path = os.path.join(in_dir, args.directors_json)

    # Validate JSON inputs
    for p in (actors_json_path, directors_json_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input JSON not found: {os.path.abspath(p)}")

    # Resolve output dirs (under in_dir)
    out_actors = os.path.join(in_dir, args.out_actors_dir)
    out_directors = os.path.join(in_dir, args.out_directors_dir)

    # Load JSONs
    print("Loading JSON files ...")
    actors = load_json(actors_json_path)
    directors = load_json(directors_json_path)

    # Download actors
    print(f"Downloading ACTOR images → {out_actors} (size={args.size}) ...")
    n_act = process_group(
        actors,
        img_dir=out_actors,
        size=args.size,
        timeout=args.timeout,
        sleep_s=args.sleep,
        skip_existing=args.skip_existing,
    )

    # Download directors
    print(f"Downloading DIRECTOR images → {out_directors} (size={args.size}) ...")
    n_dir = process_group(
        directors,
        img_dir=out_directors,
        size=args.size,
        timeout=args.timeout,
        sleep_s=args.sleep,
        skip_existing=args.skip_existing,
    )

    # Optionally update JSONs with image_url fields
    if not args.no_update_json:
        save_json(actors_json_path, actors)
        save_json(directors_json_path, directors)
        print("Updated JSON files with image_url fields.")

    # Summary
    print("\n--- Summary ---")
    print(
        {
            "in_dir": in_dir,
            "actors_json": os.path.abspath(actors_json_path),
            "directors_json": os.path.abspath(directors_json_path),
            "actors_img_dir": os.path.abspath(out_actors),
            "directors_img_dir": os.path.abspath(out_directors),
            "downloaded_actors": n_act,
            "downloaded_directors": n_dir,
            "size": args.size,
            "skip_existing": args.skip_existing,
            "updated_json": not args.no_update_json,
        }
    )


if __name__ == "__main__":
    main()
