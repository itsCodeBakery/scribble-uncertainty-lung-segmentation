#!/usr/bin/env python3
"""CASR-Net Block 03: read-only raw CT/test-split compatibility audit.

Examines the original 18-case dataset ONLY and does not generate predictions,
alter checkpoints, or overwrite pre-existing experimental artifacts.
"""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
CONFIG = ROOT / "configs" / "preprocessing_config.json"
REPORT = ROOT / "experiments" / "qualitative_comparison" / "audits" / "block03_casrnet_input_compatibility.json"
CHECKPOINT = Path("/kaggle/working/casrnet_preservation_audit/recovered_student_best_model.pth")
PREFERRED = ("coronacases_001", 139)

def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def inspect_volume(path):
    info = {"path": str(path), "exists": path.is_file()}
    if info["exists"]:
        image = nib.load(str(path))
        info.update({
            "shape": list(image.shape),
            "dtype": str(image.get_data_dtype()),
            "axis_codes": list(nib.aff2axcodes(image.affine)),
            "spacing": [float(z) for z in image.header.get_zooms()[:3]],
        })
    return info

def main():
    print("=" * 91)
    print("CASR-NET BLOCK 03 — ORIGINAL CT INPUT + TEST SPLIT COMPATIBILITY AUDIT")
    print("=" * 91)

    assert (SPLITS / "dataset_index.csv").is_file(), "Missing metadata: dataset_index.csv"
    assert (SPLITS / "test_slices.csv").is_file(), "Missing metadata: test_slices.csv"
    assert CONFIG.is_file(), "Missing original preprocessing_config.json"

    index = {x["case_id"]: x for x in rows(SPLITS / "dataset_index.csv")}
    tests = rows(SPLITS / "test_slices.csv")
    cases = sorted({x["case_id"] for x in tests})
    settings = json.loads(CONFIG.read_text())
    image_size = int(settings["image_size"])
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "andrewmvd/covid19-ct-scans",
        "checkpoint_exists": CHECKPOINT.exists(),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size if CHECKPOINT.exists() else None,
        "config": settings,
        "test_slices": len(tests),
        "test_cases": cases,
        "case_files": {},
    }

    print("\n1. Checkpoint:", "FOUND" if CHECKPOINT.exists() else "NOT FOUND")
    print("2. Original test split:", len(tests), "slices across", len(cases), "cases")
    print("3. Original preprocessing config:", settings)
    print("\n4. NIfTI source availability and image geometry:")

    for case in cases:
        meta = index.get(case)
        if not meta:
            report["case_files"][case] = {"error": "case missing in dataset_index.csv"}
            print("  ", case, "NOT IN INDEX")
            continue
        d = {k: inspect_volume(Path(meta[k]))
             for k in ("ct_path", "infection_mask_path", "lung_mask_path")}
        report["case_files"][case] = d
        good = all(v["exists"] for v in d.values())
        shapes = {k: v.get("shape") for k, v in d.items()}
        print(f"  {case}: {'ALL PRESENT' if good else 'MISSING SOURCE'}; shapes={shapes}")

    print("\n5. Compare a test slice with the archived pixel-count metadata:")
    preferred = next((r for r in tests if
                      r["case_id"] == PREFERRED[0] and
                      int(r["slice_idx"]) == PREFERRED[1]), None)
    if preferred is None:
        print("  Preferred example not found; no slice comparison performed.")
    else:
        meta = index[preferred["case_id"]]
        ct_path = Path(meta["ct_path"])
        lesion_path = Path(meta["infection_mask_path"])
        lung_path = Path(meta["lung_mask_path"])
        if all(p.is_file() for p in (ct_path, lesion_path, lung_path)):
            c, l, g = (nib.load(str(p)) for p in (ct_path, lesion_path, lung_path))
            k = int(preferred["slice_idx"])
            record = {"case": preferred["case_id"], "slice_idx": k,
                      "saved_lesion_pixels": int(preferred["infection_pixels"]),
                      "saved_lung_pixels": int(preferred["lung_pixels"]),
                      "image_shape": list(c.shape), "infection_shape": list(l.shape)}
            if all(img.ndim == 3 and img.shape[-1] > k for img in (c,l,g)):
                ct_slice = np.asanyarray(c.dataobj[:, :, k])
                lesion = np.asanyarray(l.dataobj[:, :, k])
                lung = np.asanyarray(g.dataobj[:, :, k])
                mask256 = cv2.resize((lesion > 0).astype("uint8"),
                                     (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                lung256 = cv2.resize((lung > 0).astype("uint8"),
                                     (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                record.update({
                    "ct_slice_shape": list(ct_slice.shape),
                    "ct_min_max": [float(np.min(ct_slice)), float(np.max(ct_slice))],
                    "lesion_values": [float(v) for v in np.unique(lesion)[:15]],
                    "source_lesion_pixels": int((lesion > 0).sum()),
                    "nearest_resized_lesion_pixels": int(mask256.sum()),
                    "nearest_resized_lung_pixels": int(lung256.sum()),
                    "nearest_lung_count_matches": int(lung256.sum()) == int(preferred["lung_pixels"]),
                    "nearest_lesion_count_matches": int(mask256.sum()) == int(preferred["infection_pixels"]),
                })
                print(f"  {record['case']} slice {k}: CT shape={ct_slice.shape}; "
                      f"CT intensity range={record['ct_min_max']}")
                print("  Lesion pixels (archived / source / nearest-resized):",
                      record["saved_lesion_pixels"],
                      record["source_lesion_pixels"],
                      record["nearest_resized_lesion_pixels"])
                print("  Lung pixels (archived / nearest-resized):",
                      record["saved_lung_pixels"],
                      record["nearest_resized_lung_pixels"])
                print("  Exact resized count match is only a diagnostic, not proof of the full preprocessing transform.")
            else:
                record["error"] = "unexpected NIfTI axis layout; no slice decoded"
                print("  Unexpected NIfTI axis layout; slice extraction skipped.")
            report["example_slice"] = record
        else:
            print("  NIfTI source file missing; slice comparison skipped.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nReport:", REPORT)
    print("BLOCK 03 COMPLETE — No model inference, training, or source modification.")
    print("=" * 91)

if __name__ == "__main__":
    main()
