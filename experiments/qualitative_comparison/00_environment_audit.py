#!/usr/bin/env python3
"""CASR-Net: Block 00 — read-only Kaggle input/repository/checkpoint audit.

Outputs one JSON report. Does not train, modify datasets, download CT scans,
overwrite historical experiment results, or print credentials.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "qualitative_comparison" / "audits" / "block00_environment.json"
EXPECTED = {
    "original_volumetric": "andrewmvd/covid19-ct-scans",
    "covid_lesion_71_cases": "maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset",
    "mosmed": "mathurinache/mosmeddata-chest-ct-scans-with-covid19",
}

def sh(*args):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {"ok": p.returncode == 0, "stdout": p.stdout.strip()[:1500], "stderr": p.stderr.strip()[:500]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

def inputs():
    base = Path("/kaggle/input")
    result = {"root_exists": base.exists(), "directories": [], "identified": {}}
    if base.exists():
        dirs = sorted(p for p in base.iterdir() if p.is_dir())
        # Kaggle currently uses both flat /kaggle/input/slug and the owner/slug layout.
        nested = [p for root in dirs if root.name == "datasets" for owner in root.iterdir() if owner.is_dir() for p in owner.iterdir() if p.is_dir()]
        all_dirs = dirs + nested
        result["directories"] = [str(p) for p in all_dirs[:75]]
        for name, ident in EXPECTED.items():
            slug = ident.split("/")[-1]
            match = [str(p) for p in all_dirs if p.name == slug or (slug in p.name)]
            result["identified"][name] = {"id": ident, "found": bool(match), "paths": match[:5]}
    return result

def split_audit():
    root = ROOT / "data" / "splits"
    out = {"available": root.exists(), "splits": {}, "case_overlap": {}}
    cases = {}
    for part in ("train", "val", "test"):
        fn = root / (part + "_slices.csv")
        if not fn.exists():
            out["splits"][part] = {"available": False}
            continue
        with fn.open(newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            n = 0
            cs = set()
            for row in rows:
                n += 1
                if row.get("case_id"):
                    cs.add(row["case_id"])
        cases[part] = cs
        out["splits"][part] = {"available": True, "slices": n, "cases": len(cs), "case_ids": sorted(cs)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out["case_overlap"][a + "_vs_" + b] = sorted(cases.get(a, set()) & cases.get(b, set()))
    return out

def local_checkpoints():
    ck = ROOT / "outputs" / "checkpoints"
    return {
        "directory_available_in_sparse_checkout": ck.exists(),
        "files": [{"path": str(p.relative_to(ROOT)), "size_mb": round(p.stat().st_size/1e6, 2)}
                  for p in sorted(ck.rglob("*")) if p.is_file() and p.suffix in (".pth", ".pt", ".ckpt")][:50] if ck.exists() else [],
        "note": "This notebook uses a sparse Git checkout. No local weights does not imply no weights exist in GitHub or previous Kaggle outputs."
    }

def main():
    print("=" * 88)
    print("CASR-NET BLOCK 00 — KAGGLE DATA, SPLIT, AND REPOSITORY AUDIT")
    print("=" * 88)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repository": {"root": str(ROOT), "branch": sh("git", "branch", "--show-current"),
                       "commit": sh("git", "rev-parse", "--short", "HEAD")},
        "inputs": inputs(),
        "original_split_audit": split_audit(),
        "local_checkpoints": local_checkpoints(),
        "configuration": {},
        "gpu": {}
    }
    try:
        import torch
        report["gpu"] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                         "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as exc:
        report["gpu"] = {"error": str(exc)}
    for name in ("preprocessing_config.json", "tta_postprocessing_config.json"):
        path = ROOT / "configs" / name
        if path.exists():
            report["configuration"][name] = json.loads(path.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nGPU:", report["gpu"])
    print("\nDATASETS")
    for key, d in report["inputs"]["identified"].items():
        print(f"  {key:<24} {'FOUND' if d['found'] else 'NOT ATTACHED':<12} {d['id']}")
        for p in d["paths"]:
            print("    ", p)
    print("\nORIGINAL CASE-LEVEL SPLITS")
    for key, d in report["original_split_audit"]["splits"].items():
        print(f"  {key:<6} cases={d.get('cases', 'N/A')} slices={d.get('slices', 'N/A')}")
    print("  Case overlap:", report["original_split_audit"]["case_overlap"])
    print("\nCONFIGURATIONS")
    for name, data in report["configuration"].items():
        print("  ", name, "->", json.dumps(data, sort_keys=True)[:600])
    print("\nLOCAL CHECKPOINTS:", len(report["local_checkpoints"]["files"]))
    print("  Sparse checkout may omit GitHub checkpoints; checkpoint provenance will be audited separately.")
    print("\nAudit saved:", OUT)
    print("\nBLOCK 00 COMPLETE. Share this text output before proceeding.")
    print("=" * 88)

if __name__ == "__main__":
    main()
