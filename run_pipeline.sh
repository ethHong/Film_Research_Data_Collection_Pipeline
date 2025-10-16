#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# (Optional) Load TMDB API key from a txt file
# -------------------------------
KEY_FILE="./tmdb_key.txt"
if [[ -f "${KEY_FILE}" ]]; then
  export TMDB_API_KEY="$(tr -d '[:space:]' < "${KEY_FILE}")"
  echo "Loaded TMDB_API_KEY from ${KEY_FILE}"
fi

# -------------------------------
# Step 0: Ensure spaCy model installed
# -------------------------------
python -m spacy validate | grep -q "en_core_web_sm" || python -m spacy download en_core_web_sm

# -------------------------------
# Config
# -------------------------------
IN_DIR="./data"                        # Excel source directory
export SAMPLE="${SAMPLE:-true}"        # true → load sampled Excel files, false → load full dataset
PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_DIR="wiki_bio_collected/processed"
ACTORS_CSV="actors_bio.csv"
DIRECTORS_CSV="directors_bio.csv"

# Script paths
FETCH_SCRIPT="fetch_wiki_bios.py"              # (1) Bio collection
PROCESS_SCRIPT="NLI_hint_phrase_extraction.py" # (2) Phrase extraction
TAG_SCRIPT="map_race_tags.py"                  # (3) Race tag mapping
TMDB_SCRIPT="TMDB_data_collection.py"          # (4) TMDB data collection (JSON)
TMDB_IMG_SCRIPT="TMDB_image_collection.py"     # (5) TMDB image download
FAIRFACE_SCRIPT="fairface_race_gender_detection.py" # (6) FairFace detection + inference
MERGE_SCRIPT="merge_wiki_tmdb_fairface.py"     # (7) Merge all sources
FINAL_SCRIPT="final_backfill.py"               # (8) Final backfill + aggregates

# TMDB artifacts
TMDB_JSON_DIR="TMDB_data_collected"            # where JSONs are saved
TMDB_IMG_ACTORS_SUBDIR="img_actors"            # subdir under TMDB_JSON_DIR
TMDB_IMG_DIRECTORS_SUBDIR="img_directors"      # subdir under TMDB_JSON_DIR

# Image download options
TMDB_IMG_SIZE="${TMDB_IMG_SIZE:-w500}"         # override with env if you want
TMDB_IMG_SKIP_EXISTING="${TMDB_IMG_SKIP_EXISTING:-true}" # skip downloading if file exists

# FairFace output
FAIRFACE_OUT_DIR="fairface_result"
FAIRFACE_CSV="${FAIRFACE_OUT_DIR}/race_prediction.csv"

# Final outputs
FINAL_OUT_DIR="Final_Output"

# Flag for sample/full (propagate to scripts that support it)
FULL_FLAG=""
shopt -s nocasematch
if [[ "${SAMPLE}" == "false" || "${SAMPLE}" == "0" || "${SAMPLE}" == "no" || "${SAMPLE}" == "n" ]]; then
  FULL_FLAG="--full"
fi
shopt -u nocasematch

# -------------------------------
# Pre-flight checks
# -------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python not found. Set PYTHON_BIN or ensure python is on PATH."
  exit 1
fi

if [[ ! -d "${IN_DIR}" ]]; then
  echo "ERROR: IN_DIR not found: ${IN_DIR}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

# -------------------------------
# (1) Run: fetch_wiki_bios.py
# -------------------------------
echo "[1/8] Running fetch_wiki_bios.py ..."
if [[ -n "${FULL_FLAG}" ]]; then
  "${PYTHON_BIN}" "${FETCH_SCRIPT}" --path "${IN_DIR}"
else
  "${PYTHON_BIN}" "${FETCH_SCRIPT}" --path "${IN_DIR}" --sample
fi

if [[ ! -f "wiki_bio_collected/${ACTORS_CSV}" ]] || [[ ! -f "wiki_bio_collected/${DIRECTORS_CSV}" ]]; then
  echo "ERROR: Missing expected output CSVs from step (1)."
  exit 1
fi

# -------------------------------
# (2) Run: NLI_hint_phrase_extraction.py
# -------------------------------
echo "[2/8] Running NLI_hint_phrase_extraction.py ..."
"${PYTHON_BIN}" "${PROCESS_SCRIPT}" \
  --in_dir "wiki_bio_collected" \
  --out_dir "${OUT_DIR}" \
  --actors_file "${ACTORS_CSV}" \
  --directors_file "${DIRECTORS_CSV}"

if [[ ! -f "${OUT_DIR}/actors_bio_hint_phrases_processed_final_merged.csv" ]] || \
   [[ ! -f "${OUT_DIR}/directors_bio_hint_phrases_processed_final_merged.csv" ]]; then
  echo "ERROR: Processed files not found after step (2)."
  exit 1
fi

# -------------------------------
# (3) Run: map_race_tags.py
# -------------------------------
echo "[3/8] Running map_race_tags.py ..."
"${PYTHON_BIN}" "${TAG_SCRIPT}" --in_dir "wiki_bio_collected"

if [[ ! -f "wiki_bio_collected/output/actors_tag_from_wiki.csv" ]] || \
   [[ ! -f "wiki_bio_collected/output/director_tag_from_wiki.csv" ]]; then
  echo "ERROR: Final wiki-tag CSVs not found in wiki_bio_collected/output/"
  exit 1
fi

# -------------------------------
# (4) Run: TMDB_data_collection.py  (JSON writer)
# -------------------------------
echo "[4/8] Running TMDB_data_collection.py ..."
mkdir -p "${TMDB_JSON_DIR}"

if [[ -n "${TMDB_API_KEY:-}" ]]; then
  # If your TMDB script supports --api_key, you can add it here; it also can read env var.
  "${PYTHON_BIN}" "${TMDB_SCRIPT}" \
    --path "${IN_DIR}" \
    --out_dir "${TMDB_JSON_DIR}" \
    ${FULL_FLAG}
else
  echo "WARNING: TMDB_API_KEY not set. Ensure TMDB_data_collection.py can read API key (env or config)."
  "${PYTHON_BIN}" "${TMDB_SCRIPT}" \
    --path "${IN_DIR}" \
    --out_dir "${TMDB_JSON_DIR}" \
    ${FULL_FLAG}
fi

if [[ ! -f "${TMDB_JSON_DIR}/actor_TMDB_data.json" ]] || \
   [[ ! -f "${TMDB_JSON_DIR}/director_TMDB_data.json" ]]; then
  echo "ERROR: TMDB JSONs not found in ${TMDB_JSON_DIR}/"
  exit 1
fi

# -------------------------------
# (5) Run: TMDB_image_collection.py  (image downloader)
# -------------------------------
echo "[5/8] Running TMDB_image_collection.py ..."
mkdir -p "${TMDB_JSON_DIR}/${TMDB_IMG_ACTORS_SUBDIR}" "${TMDB_JSON_DIR}/${TMDB_IMG_DIRECTORS_SUBDIR}"

IMG_FLAGS=()
if [[ "${TMDB_IMG_SKIP_EXISTING}" == "true" ]]; then
  IMG_FLAGS+=(--skip_existing)
fi

"${PYTHON_BIN}" "${TMDB_IMG_SCRIPT}" \
  --in_dir "${TMDB_JSON_DIR}" \
  --actors_json "actor_TMDB_data.json" \
  --directors_json "director_TMDB_data.json" \
  --out_actors_dir "${TMDB_IMG_ACTORS_SUBDIR}" \
  --out_directors_dir "${TMDB_IMG_DIRECTORS_SUBDIR}" \
  --size "${TMDB_IMG_SIZE}" \
  "${IMG_FLAGS[@]}"

# Basic verification (at least one image)
if ! compgen -G "${TMDB_JSON_DIR}/${TMDB_IMG_ACTORS_SUBDIR}/*" > /dev/null && \
   ! compgen -G "${TMDB_JSON_DIR}/${TMDB_IMG_DIRECTORS_SUBDIR}/*" > /dev/null; then
  echo "WARNING: No images found in ${TMDB_JSON_DIR}/{${TMDB_IMG_ACTORS_SUBDIR},${TMDB_IMG_DIRECTORS_SUBDIR}}"
else
  echo "Images saved under:"
  echo "  - ${TMDB_JSON_DIR}/${TMDB_IMG_ACTORS_SUBDIR}"
  echo "  - ${TMDB_JSON_DIR}/${TMDB_IMG_DIRECTORS_SUBDIR}"
fi

# -------------------------------
# (6) Run: fairface_race_gender_detection.py (face crop + predict)
# -------------------------------
echo "[6/8] Running fairface_race_gender_detection.py ..."
mkdir -p "${FAIRFACE_OUT_DIR}"

# Optional model checks (helpful errors)
if [[ ! -f "dlib_models/mmod_human_face_detector.dat" ]] || [[ ! -f "dlib_models/shape_predictor_5_face_landmarks.dat" ]]; then
  echo "ERROR: Missing dlib model files under dlib_models/. Please place:"
  echo "  - dlib_models/mmod_human_face_detector.dat"
  echo "  - dlib_models/shape_predictor_5_face_landmarks.dat"
  exit 1
fi
if [[ ! -f "fair_face_models/res34_fair_align_multi_7_20190809.pt" ]] || [[ ! -f "fair_face_models/res34_fair_align_multi_4_20190809.pt" ]]; then
  echo "ERROR: Missing FairFace model files under fair_face_models/. Please place:"
  echo "  - fair_face_models/res34_fair_align_multi_7_20190809.pt"
  echo "  - fair_face_models/res34_fair_align_multi_4_20190809.pt"
  exit 1
fi

"${PYTHON_BIN}" "${FAIRFACE_SCRIPT}" \
  --out_dir "${FAIRFACE_OUT_DIR}"

if [[ ! -f "${FAIRFACE_CSV}" ]]; then
  echo "ERROR: FairFace CSV not found: ${FAIRFACE_CSV}"
  exit 1
fi

# -------------------------------
# (7) Run: merge_wiki_tmdb_fairface.py
# -------------------------------
echo "[7/8] Running merge_wiki_tmdb_fairface.py ..."
MERGE_OUT="film_industry_complete_data.csv"
if [[ -n "${TMDB_API_KEY:-}" ]]; then
  "${PYTHON_BIN}" "${MERGE_SCRIPT}" \
    --tmdb_dir "${TMDB_JSON_DIR}" \
    --wiki_dir "wiki_bio_collected/output" \
    --data_dir "${IN_DIR}" \
    --fairface_csv "${FAIRFACE_CSV}" \
    --out_csv "${MERGE_OUT}" \
    --api_key "${TMDB_API_KEY}" \
    ${FULL_FLAG}
else
  "${PYTHON_BIN}" "${MERGE_SCRIPT}" \
    --tmdb_dir "${TMDB_JSON_DIR}" \
    --wiki_dir "wiki_bio_collected/output" \
    --data_dir "${IN_DIR}" \
    --fairface_csv "${FAIRFACE_CSV}" \
    --out_csv "${MERGE_OUT}" \
    ${FULL_FLAG}
fi

if [[ ! -f "${MERGE_OUT}" ]]; then
  echo "ERROR: Merge output not found: ${MERGE_OUT}"
  exit 1
fi

# -------------------------------
# (8) Run: final_backfill.py
# -------------------------------
echo "[8/8] Running final_backfill.py ..."
mkdir -p "${FINAL_OUT_DIR}"

"${PYTHON_BIN}" "${FINAL_SCRIPT}" \
  --out_dir "${FINAL_OUT_DIR}" \
  ${FULL_FLAG}

# Verify final outputs (at least the aggregate tables)
if ! compgen -G "${FINAL_OUT_DIR}/*movie_ppl_*aggregate*.csv" > /dev/null; then
  echo "WARNING: No final aggregate CSVs found in ${FINAL_OUT_DIR}/"
else
  echo "Final outputs saved in ${FINAL_OUT_DIR}/"
fi

# -------------------------------
# Done
# -------------------------------
echo "✅ All steps completed successfully!"
echo "Outputs saved in:"
echo "  - ${OUT_DIR}/"
echo "  - wiki_bio_collected/output/actors_tag_from_wiki.csv"
echo "  - wiki_bio_collected/output/director_tag_from_wiki.csv"
echo "  - ${TMDB_JSON_DIR}/actor_TMDB_data.json"
echo "  - ${TMDB_JSON_DIR}/director_TMDB_data.json"
echo "  - ${TMDB_JSON_DIR}/${TMDB_IMG_ACTORS_SUBDIR}/"
echo "  - ${TMDB_JSON_DIR}/${TMDB_IMG_DIRECTORS_SUBDIR}/"
echo "  - ${FAIRFACE_CSV}"
echo "  - ${MERGE_OUT}"
echo "  - ${FINAL_OUT_DIR}/"
