#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Clean only PIPELINE-GENERATED artifacts, keep directories.
# NEVER touch Final_Output (protected).
# Usage:
#   ./clean_pipeline_outputs.sh            # dry-run (default)
#   ./clean_pipeline_outputs.sh --force    # actually delete
#   SAMPLE=false ./clean_pipeline_outputs.sh --force
# ============================================================

DRY_RUN=true
VERBOSE=false
for arg in "$@"; do
  case "$arg" in
    --force) DRY_RUN=false ;;
    --dry-run) DRY_RUN=true ;;
    -v|--verbose) VERBOSE=true ;;
    *) echo "Unknown arg: $arg" ; exit 2 ;;
  esac
done

log() { echo -e "$@"; }
vlog() { $VERBOSE && log "$@"; }

# ---------- Config (align with run_pipeline.sh) ----------
IN_DIR="${IN_DIR:-./data}"
OUT_DIR="${OUT_DIR:-wiki_bio_collected/processed}"
TMDB_JSON_DIR="${TMDB_JSON_DIR:-TMDB_data_collected}"
TMDB_IMG_ACTORS_SUBDIR="${TMDB_IMG_ACTORS_SUBDIR:-img_actors}"
TMDB_IMG_DIRECTORS_SUBDIR="${TMDB_IMG_DIRECTORS_SUBDIR:-img_directors}"
FAIRFACE_OUT_DIR="${FAIRFACE_OUT_DIR:-fairface_result}"
FAIRFACE_CSV="${FAIRFACE_CSV:-${FAIRFACE_OUT_DIR}/race_prediction.csv}"
# PROTECTED: Do not touch Final_Output
FINAL_OUT_DIR="${FINAL_OUT_DIR:-Final_Output}"
MERGE_OUT="${MERGE_OUT:-film_industry_complete_data.csv}"

# Some known outputs produced by the pipeline
WIKI_TAG_ACTORS="${WIKI_TAG_ACTORS:-wiki_bio_collected/output/actors_tag_from_wiki.csv}"
WIKI_TAG_DIRECTORS="${WIKI_TAG_DIRECTORS:-wiki_bio_collected/output/director_tag_from_wiki.csv}"

# Optional: FairFace intermediate crops dir (if your pipeline created it)
CROPPED_FACES_DIR="${CROPPED_FACES_DIR:-cropped_faces}"

# ---------- Safety checks ----------
confirm_repo_root() {
  # Basic sanity check to avoid "/" or home directory disasters
  for p in "." "$OUT_DIR" "$TMDB_JSON_DIR" "$FAIRFACE_OUT_DIR"; do
    if [[ "$p" == "/" || "$p" == "$HOME" ]]; then
      echo "Refusing to run: dangerous path detected ($p)"; exit 3
    fi
  done
}
confirm_repo_root

# Hard protection: ensure we never include Final_Output
protect_final_output() {
  for arg in "$@"; do
    if [[ "$arg" == "$FINAL_OUT_DIR" || "$arg" == "$FINAL_OUT_DIR/"* ]]; then
      echo "Refusing to operate on protected path: $arg"; exit 4
    fi
  done
}

# ---------- Deletion helpers ----------
delete_file() {
  local f="$1"
  protect_final_output "$f"
  if [[ -f "$f" || -L "$f" ]]; then
    if $DRY_RUN; then
      log "DRY-RUN: delete file: $f"
    else
      rm -f -- "$f"
      vlog "Deleted file: $f"
    fi
  else
    vlog "skip (not found): $f"
  fi
}

# Delete files under a directory (recursively), KEEP directories
clear_dir_files() {
  local d="$1"
  protect_final_output "$d"
  if [[ -d "$d" ]]; then
    if $DRY_RUN; then
      # Preview files that would be deleted
      find "$d" -type f -print | sed 's/^/DRY-RUN: delete file: /'
    else
      # Remove all files under d (recursively), leave directories intact
      find "$d" -type f -print -delete
      vlog "Cleared files under: $d"
    fi
  else
    vlog "skip (dir not found): $d"
  fi
}

# ---------- Targets (files & directories) ----------
# NOTE: Final_Output is intentionally NOT listed anywhere below.
FILES_TO_DELETE=(
  "$WIKI_TAG_ACTORS"
  "$WIKI_TAG_DIRECTORS"
  "$TMDB_JSON_DIR/actor_TMDB_data.json"
  "$TMDB_JSON_DIR/director_TMDB_data.json"
  "$FAIRFACE_CSV"
  "$MERGE_OUT"
  "wiki_bio_collected/actors_bio.csv"        
  "wiki_bio_collected/directors_bio.csv" 
)

DIRS_TO_CLEAR=(
  "$OUT_DIR"
  "$TMDB_JSON_DIR/$TMDB_IMG_ACTORS_SUBDIR"
  "$TMDB_JSON_DIR/$TMDB_IMG_DIRECTORS_SUBDIR"
  "$FAIRFACE_OUT_DIR"
  "$CROPPED_FACES_DIR"
)

# ---------- Execute ----------
log "==> Cleaning pipeline-generated artifacts (directories preserved)"
$DRY_RUN && log "(dry-run mode; use --force to actually delete)\n"

for f in "${FILES_TO_DELETE[@]}"; do
  delete_file "$f"
done

for d in "${DIRS_TO_CLEAR[@]}"; do
  clear_dir_files "$d"
done

log "\nDone."
