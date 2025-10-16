#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect faces (dlib), crop aligned faces, and run FairFace race/gender/age prediction.

Inputs:
  - --in_dir: Base directory containing TMDB images (default: TMDB_data_collected)
  - --actors_dir / --directors_dir: Subdirectories for actor/director images
    (default: img_actors / img_directors)
  - dlib models in --dlib_dir:
      mmod_human_face_detector.dat
      shape_predictor_5_face_landmarks.dat
  - FairFace model weights in --fairface_dir:
      res34_fair_align_multi_7_20190809.pt
      res34_fair_align_multi_4_20190809.pt

Outputs (under --out_dir, default: fairface_result):
  - input_imgs.csv               # auto-generated list of image paths
  - detected_faces/              # cropped aligned faces by dlib
  - race_prediction.csv          # FairFace predictions on cropped faces

Notes:
  - Comments in English only (per request).
"""

from __future__ import print_function, division

import os
import re
import json
import argparse
import warnings
from typing import List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# dlib / torch / torchvision
import dlib
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


# ---------------------------
# Argparse
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Face detection (dlib) + FairFace race/gender/age prediction"
    )
    # IO roots
    parser.add_argument(
        "--in_dir",
        type=str,
        default="TMDB_data_collected",
        help="Base directory containing TMDB images.",
    )
    parser.add_argument(
        "--actors_dir",
        type=str,
        default="img_actors",
        help="Subdirectory of in_dir containing actor images.",
    )
    parser.add_argument(
        "--directors_dir",
        type=str,
        default="img_directors",
        help="Subdirectory of in_dir containing director images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="fairface_result",
        help="Directory to write input CSV, crops, and predictions.",
    )

    # Model directories
    parser.add_argument(
        "--dlib_dir",
        type=str,
        default="dlib_models",
        help="Directory containing dlib model .dat files.",
    )
    parser.add_argument(
        "--fairface_dir",
        type=str,
        default="fair_face_models",
        help="Directory containing FairFace .pt weights.",
    )

    # File names inside out_dir
    parser.add_argument(
        "--csv_name",
        type=str,
        default="input_imgs.csv",
        help="Generated CSV filename listing input image paths.",
    )
    parser.add_argument(
        "--detected_subdir",
        type=str,
        default="detected_faces",
        help="Subdirectory for cropped faces under out_dir.",
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="race_prediction.csv",
        help="Output CSV with predictions (under out_dir).",
    )

    # Detection params
    parser.add_argument(
        "--default_max_size",
        type=int,
        default=800,
        help="Resize larger edge to this before detection.",
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=300,
        help="Aligned face chip size for dlib.get_face_chips.",
    )
    parser.add_argument(
        "--padding", type=float, default=0.25, help="Padding for dlib.get_face_chips."
    )

    # Runtime
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Force CUDA if available (otherwise auto-select).",
    )

    return parser.parse_args()


# ---------------------------
# Utils
# ---------------------------
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def list_images(dir_path: str) -> List[str]:
    """List image files (jpg/jpeg/png) with full paths."""
    if not os.path.isdir(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(dir_path, name))
    return sorted(files)


# ---------------------------
# Dlib helpers
# ---------------------------
def detect_and_align_faces(
    image_paths: List[str],
    save_dir: str,
    dlib_dir: str,
    default_max_size: int = 800,
    chip_size: int = 300,
    padding: float = 0.25,
):
    """Run dlib CNN face detection + 5-landmark alignment, save aligned crops."""
    ensure_dir(save_dir)

    det_path = os.path.join(dlib_dir, "mmod_human_face_detector.dat")
    sp_path = os.path.join(dlib_dir, "shape_predictor_5_face_landmarks.dat")
    if not os.path.exists(det_path) or not os.path.exists(sp_path):
        raise FileNotFoundError(
            f"Missing dlib models. Expected:\n  - {det_path}\n  - {sp_path}"
        )

    cnn_face_detector = dlib.cnn_face_detection_model_v1(det_path)
    sp = dlib.shape_predictor(sp_path)

    for idx, image_path in enumerate(tqdm(image_paths, desc="Detecting faces")):
        try:
            img = dlib.load_rgb_image(image_path)
        except Exception:
            print(f"[WARN] Failed to read: {image_path}")
            continue

        # Resize while preserving aspect ratio
        h, w = img.shape[:2]
        if w > h:
            new_w, new_h = default_max_size, int(default_max_size * h / w)
        else:
            new_w, new_h = int(default_max_size * w / h), default_max_size
        img_resized = dlib.resize_image(img, rows=new_h, cols=new_w)

        # Detect faces
        dets = cnn_face_detector(img_resized, 1)
        if len(dets) == 0:
            # No face found for this image
            continue

        # 5-point landmark alignment
        faces = dlib.full_object_detections()
        for d in dets:
            faces.append(sp(img_resized, d.rect))

        chips = dlib.get_face_chips(img_resized, faces, size=chip_size, padding=padding)

        # Save each chip
        base = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(os.path.basename(image_path))[1]
        for fi, chip in enumerate(chips):
            out_name = f"{base}_face{fi}{ext.lower() or '.jpg'}"
            out_path = os.path.join(save_dir, out_name)
            dlib.save_image(chip, out_path)


# ---------------------------
# FairFace prediction
# ---------------------------
def load_fairface_models(fairface_dir: str, device: torch.device):
    """Load FairFace 7-class and 4-class resnet34 heads."""
    # 7-class head (race + gender + age)
    m7 = torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.DEFAULT
    )
    m7.fc = nn.Linear(m7.fc.in_features, 18)
    w7 = os.path.join(fairface_dir, "res34_fair_align_multi_7_20190809.pt")
    if not os.path.exists(w7):
        raise FileNotFoundError(f"Missing FairFace weight: {w7}")
    m7.load_state_dict(torch.load(w7, map_location="cpu"))
    m7 = m7.to(device).eval()

    # 4-class head (race-4)
    m4 = torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.DEFAULT
    )
    m4.fc = nn.Linear(m4.fc.in_features, 18)
    w4 = os.path.join(fairface_dir, "res34_fair_align_multi_4_20190809.pt")
    if not os.path.exists(w4):
        raise FileNotFoundError(f"Missing FairFace weight: {w4}")
    m4.load_state_dict(torch.load(w4, map_location="cpu"))
    m4 = m4.to(device).eval()

    return m7, m4


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax for 1D array."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def predict_on_crops(
    crops_dir: str,
    out_csv_path: str,
    fairface_dir: str,
    device: torch.device,
):
    """Run FairFace on all cropped faces and save a CSV with predictions."""
    crop_files = list_images(crops_dir)
    if len(crop_files) == 0:
        # Still write an empty CSV with headers to keep pipeline stable
        pd.DataFrame(
            columns=[
                "face_name_align",
                "race",
                "race4",
                "gender",
                "age",
                "race_scores_fair",
                "race_scores_fair_4",
                "gender_scores_fair",
                "age_scores_fair",
            ]
        ).to_csv(out_csv_path, index=False)
        print(f"[WARN] No crops found in {crops_dir}. Wrote empty CSV.")
        return

    # Load models once
    model_fair_7, model_fair_4 = load_fairface_models(fairface_dir, device)

    # Preprocess
    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Storage
    face_names = []
    race_scores_fair, race_preds_fair = [], []
    gender_scores_fair, gender_preds_fair = [], []
    age_scores_fair, age_preds_fair = [], []
    race_scores_fair_4, race_preds_fair_4 = [], []

    for img_name in tqdm(crop_files, desc="Predicting age/gender/race"):
        try:
            image = dlib.load_rgb_image(img_name)
        except Exception:
            print(f"[WARN] Failed to read crop: {img_name}")
            continue

        face_names.append(img_name)

        # to tensor
        t = trans(image).unsqueeze(0).to(device)

        # 7-class head
        with torch.no_grad():
            out7 = model_fair_7(t).cpu().numpy().squeeze()

        race7 = softmax_np(out7[:7])
        gender7 = softmax_np(out7[7:9])
        age7 = softmax_np(out7[9:18])

        race_scores_fair.append(race7.tolist())
        gender_scores_fair.append(gender7.tolist())
        age_scores_fair.append(age7.tolist())

        race_preds_fair.append(int(np.argmax(race7)))
        gender_preds_fair.append(int(np.argmax(gender7)))
        age_preds_fair.append(int(np.argmax(age7)))

        # 4-class head
        with torch.no_grad():
            out4 = model_fair_4(t).cpu().numpy().squeeze()
        race4 = softmax_np(out4[:4])
        race_scores_fair_4.append(race4.tolist())
        race_preds_fair_4.append(int(np.argmax(race4)))

    # Build DataFrame
    result = pd.DataFrame(
        {
            "face_name_align": face_names,
            "race_preds_fair": race_preds_fair,
            "race_preds_fair_4": race_preds_fair_4,
            "gender_preds_fair": gender_preds_fair,
            "age_preds_fair": age_preds_fair,
            "race_scores_fair": race_scores_fair,
            "race_scores_fair_4": race_scores_fair_4,
            "gender_scores_fair": gender_scores_fair,
            "age_scores_fair": age_scores_fair,
        }
    )

    # Map indices to labels (FairFace)
    race7_map = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East Asian",
        "Southeast Asian",
        "Indian",
        "Middle Eastern",
    ]
    race4_map = ["White", "Black", "Asian", "Indian"]
    gender_map = ["Male", "Female"]
    age_map = [
        "0-2",
        "3-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+",
    ]

    result["race"] = [
        race7_map[i] if i is not None else None for i in result["race_preds_fair"]
    ]
    result["race4"] = [
        race4_map[i] if i is not None else None for i in result["race_preds_fair_4"]
    ]
    result["gender"] = [
        gender_map[i] if i is not None else None for i in result["gender_preds_fair"]
    ]
    result["age"] = [
        age_map[i] if i is not None else None for i in result["age_preds_fair"]
    ]

    # Final CSV
    cols = [
        "face_name_align",
        "race",
        "race4",
        "gender",
        "age",
        "race_scores_fair",
        "race_scores_fair_4",
        "gender_scores_fair",
        "age_scores_fair",
    ]
    result[cols].to_csv(out_csv_path, index=False)
    print(f"Saved predictions to {out_csv_path}")


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Resolve paths
    in_dir = args.in_dir.rstrip("/")
    actors_path = os.path.join(in_dir, args.actors_dir)
    directors_path = os.path.join(in_dir, args.directors_dir)

    out_dir = args.out_dir.rstrip("/")
    ensure_dir(out_dir)

    # Build image list and write CSV
    actors_imgs = list_images(actors_path)
    directors_imgs = list_images(directors_path)
    all_imgs = actors_imgs + directors_imgs

    input_csv_path = os.path.join(out_dir, args.csv_name)
    pd.DataFrame({"img_path": all_imgs}).to_csv(input_csv_path, index=False)
    print(f"Wrote input CSV with {len(all_imgs)} images → {input_csv_path}")

    # dlib CUDA toggle
    if args.use_cuda and dlib.DLIB_USE_CUDA is False:
        # This flag is compiled at build time; we just announce availability
        print("[INFO] dlib was not compiled with CUDA; falling back to CPU.")
    print(f"Using dlib CUDA?: {dlib.DLIB_USE_CUDA}")

    # Detect and align
    crops_dir = os.path.join(out_dir, args.detected_subdir)
    ensure_dir(crops_dir)
    detect_and_align_faces(
        all_imgs,
        save_dir=crops_dir,
        dlib_dir=args.dlib_dir,
        default_max_size=args.default_max_size,
        chip_size=args.chip_size,
        padding=args.padding,
    )
    print(f"Detected faces are saved at {crops_dir}")

    # Device for torch
    device = torch.device(
        "cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        print("[INFO] Using CUDA for FairFace inference.")
    else:
        print("[INFO] Using CPU for FairFace inference.")

    # Predict
    pred_csv_path = os.path.join(out_dir, args.pred_csv)
    predict_on_crops(
        crops_dir=crops_dir,
        out_csv_path=pred_csv_path,
        fairface_dir=args.fairface_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
