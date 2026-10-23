#!/usr/bin/env python3
"""Block 04: one-slice CASR-Net recovery checkpoint inference (diagnostic only).

Preserves historical outputs. No training, no baseline inspection, no hyperparameter
search. A recovered qualitative checkpoint is NOT the final 0.8046 model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path("/kaggle/working/casrnet_preservation_audit/recovered_student_best_model.pth")
OUT = ROOT / "experiments" / "qualitative_comparison" / "outputs" / "casrnet_recovered_pilot"
CASE, SLICE = "coronacases_001", 139
THRESHOLD = 0.035  # Archived recovered-qualitative example; NOT re-optimized on test data
EXPECTED_SHA256 = "97a6166d3b222187b90312929588196bf6797240b659d37eca93f8f49a06a020"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def load_row(csv_path, **match):
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(str(row.get(k)) == str(v) for k, v in match.items()):
                return row
    raise ValueError(f"No row matching {match} in {csv_path}")

def flip(x, axes):
    return torch.flip(x, dims=axes) if axes else x

def dice_iou(pred, gt):
    p, y = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, y).sum())
    denom_d = int(p.sum() + y.sum())
    union = int(np.logical_or(p, y).sum())
    return ((2 * tp / denom_d) if denom_d else 1.0,
            (tp / union) if union else 1.0)

def main():
    print("=" * 90)
    print("CASR-NET BLOCK 04 — RECOVERED STUDENT ONE-SLICE PILOT INFERENCE")
    print("=" * 90)

    assert CKPT.is_file(), "Run checkpoint preservation Block 01 first."
    digest = sha256(CKPT)
    assert digest == EXPECTED_SHA256, "Checkpoint SHA256 differs from Block 01!"
    print("1. Verified recovered checkpoint SHA256:", digest)

    cfg = json.loads((ROOT / "configs/preprocessing_config.json").read_text())
    size = int(cfg["image_size"])
    lo, hi = float(cfg["hu_clip_min"]), float(cfg["hu_clip_max"])
    original = load_row(ROOT / "data/splits/dataset_index.csv", case_id=CASE)
    split = load_row(ROOT / "data/splits/test_slices.csv",
                     case_id=CASE, slice_idx=SLICE)
    archived = load_row(ROOT / "outputs/reports/CASRNet_Qualitative_Recovered_5x5_metadata.csv",
                        image_path=f"data/processed/images/{CASE}_slice_{SLICE:04d}.npy")

    paths = {k: Path(original[k]) for k in
             ("ct_path", "infection_mask_path", "lung_mask_path")}
    for k, p in paths.items():
        assert p.is_file(), f"Missing source for {k}: {p}"

    # Extract exactly the native axial slice specified by the archived split.
    ct_raw = np.asarray(nib.load(str(paths["ct_path"])).dataobj[:, :, SLICE],
                        dtype=np.float32)
    gt_native = np.asarray(
        nib.load(str(paths["infection_mask_path"])).dataobj[:, :, SLICE]) > 0
    lung_native = np.asarray(
        nib.load(str(paths["lung_mask_path"])).dataobj[:, :, SLICE]) > 0

    # Hypothesis consistent with the archived preprocessing settings.
    # This is a recovery diagnostic, not an assertion that the historic code
    # used the identical CT interpolation/normalization sequence.
    ct_clip = np.clip(ct_raw, lo, hi)
    ct_norm = (ct_clip - lo) / (hi - lo)
    img = cv2.resize(ct_norm, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gt = cv2.resize(gt_native.astype(np.uint8), (size, size),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    lung = cv2.resize(lung_native.astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

    print(f"2. Original {CASE} slice {SLICE}, native CT: {ct_raw.shape} -> {img.shape}")
    print(f"   Saved native lesion pixels={split['infection_pixels']}; "
          f"recovered native={int(gt_native.sum())}; "
          f"recovered 256x256={int(gt.sum())}; "
          f"archived qualitative GT pixels={archived['gt_pixels']}")
    assert int(gt_native.sum()) == int(split["infection_pixels"]), (
        "Native lesion mask and case-split metadata disagree."
    )
    assert int(gt.sum()) == int(archived["gt_pixels"]), (
        "Recovered 256x256 lesion mask does NOT match the archived qualitative example."
    )

    # The recovered qualitative checkpoint has a three-channel encoder stem.
    # It was recovered from teacher-TTA soft targets, so three repeated
    # grayscale channels are the explicit 2-D diagnostic hypothesis.
    x = torch.from_numpy(np.repeat(img[None, :, :], 3, axis=0)[None]).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("3. Device:", device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    state = ck["model_state_dict"]
    print("   Checkpoint:", ck.get("model_name"), "epoch", ck.get("epoch"))
    print("   Segmentation Models PyTorch:", getattr(smp, "__version__", "unknown"))
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights=None,
                            in_channels=3, classes=1, activation=None)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        print("\nARCHITECTURE REPRODUCTION BLOCKED:")
        print(str(exc)[:3500])
        raise RuntimeError("Do not perform inference with partially loaded weights.") from exc

    model = model.to(device).eval()
    x = x.to(device)
    flips = [(), (-1,), (-2,), (-2, -1)]
    predicted = []
    with torch.inference_mode():
        for dims in tqdm(flips, desc="CASR-Net four-view TTA", leave=False):
            xt = flip(x, dims)
            logits = model(xt)
            pr = torch.sigmoid(logits)
            predicted.append(flip(pr, dims).float().cpu().numpy()[0, 0])

    raw_prob = predicted[0]
    tta_prob = np.mean(np.stack(predicted), axis=0).astype(np.float32)

    # Reference recovered qualitative convention: TTA and lung restriction,
    # fixed threshold as recorded in its saved metadata. Keep separate
    # unmasked and masked diagnostics to detect preprocessing differences.
    pred_unmasked = tta_prob >= THRESHOLD
    pred = np.logical_and(pred_unmasked, lung)
    d, j = dice_iou(pred, gt)
    d_nomask, j_nomask = dice_iou(pred_unmasked, gt)
    saved_d = float(archived["dice_internal_selection_only"])
    saved_j = float(archived["iou_internal_selection_only"])
    saved_pixels = int(archived["pred_pixels"])

    print("\n4. Single-slice diagnostic (NOT a final test-set evaluation):")
    print("   Exact archived threshold:", THRESHOLD)
    print(f"   Lung-restricted -> Dice={d:.6f}, IoU={j:.6f}, "
          f"pred_pixels={int(pred.sum())}")
    print(f"   No lung restriction -> Dice={d_nomask:.6f}, IoU={j_nomask:.6f}, "
          f"pred_pixels={int(pred_unmasked.sum())}")
    print(f"   Archived recovered figure -> Dice={saved_d:.6f}, "
          f"IoU={saved_j:.6f}, pred_pixels={saved_pixels}")
    print("   NOTE: do not modify threshold/preprocessing based on this test slice.")
    print("   A difference indicates incomplete historical inference provenance.")

    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{CASE}_slice_{SLICE:04d}"
    np.savez_compressed(OUT / f"{name}_diagnostic.npz", ct=img, gt=gt.astype(np.uint8),
                        lung=lung.astype(np.uint8), raw_probability=raw_prob,
                        tta_probability=tta_prob, prediction=pred.astype(np.uint8))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    panels = [(img, "CT image", "gray"),
              (gt, "Reference infection", "gray"),
              (tta_prob, "Recovered student probability", "viridis"),
              (img, f"Student prediction (Dice {d:.3f})", "gray")]
    for ax, (view, title, cmap) in zip(axes, panels):
        ax.imshow(view, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    axes[-1].contour(gt.astype(float), levels=[0.5], colors=["lime"], linewidths=1.0)
    axes[-1].contour(pred.astype(float), levels=[0.5], colors=["red"], linewidths=1.0)
    fig.suptitle("CASR-Net recovered-checkpoint pilot — NOT a final manuscript figure",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / f"{name}_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT), "sha256": digest,
        "checkpoint_name": ck.get("model_name"), "checkpoint_epoch": ck.get("epoch"),
        "case_id": CASE, "slice_index": SLICE,
        "input_mode_hypothesis": "2D HU clip [-1000,400], minmax [0,1], 3 repeated channels",
        "spatial_size": [size, size], "tta": ["identity","horizontal","vertical","both"],
        "threshold_from_archived_qualitative_example": THRESHOLD,
        "lung_constraint_applied": True,
        "gt_pixels": int(gt.sum()), "pred_pixels": int(pred.sum()),
        "dice_pilot": d, "iou_pilot": j,
        "dice_without_lung_constraint": d_nomask, "iou_without_lung_constraint": j_nomask,
        "archived_recovered_dice": saved_d, "archived_recovered_iou": saved_j,
        "archived_recovered_pred_pixels": saved_pixels,
        "note": "Diagnostic reconstruction, not proof of original final model or preprocessing."
    }
    (OUT / f"{name}_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nFiles saved only under:", OUT)
    print("BLOCK 04 COMPLETE. Share full output before the next step.")
    print("=" * 90)

if __name__ == "__main__":
    main()
