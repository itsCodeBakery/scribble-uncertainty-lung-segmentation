#!/usr/bin/env python3
"""Block 06 — Three-row journal-style figure from verified CASR-Net .npz outputs.

No inference, threshold tuning, model fitting, mask editing, or baseline evaluation.
Preserves five-row original. Uses one predeclared slice from three distinct cases.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "qualitative_comparison" / "outputs" / "casrnet_recovered_five_examples"
DST = ROOT / "experiments" / "qualitative_comparison" / "outputs" / "casrnet_three_row_pilot"
SELECTED = (
    ("radiopaedia_7", 32),
    ("coronacases_009", 120),
    ("coronacases_001", 125),
)
CHECKPOINT_SHA256 = "97a6166d3b222187b90312929588196bf6797240b659d37eca93f8f49a06a020"

def metrics(p, g):
    a = p.astype(bool)
    b = g.astype(bool)
    t = int((a & b).sum())
    d = int(a.sum()) + int(b.sum())
    union = int((a | b).sum())
    return (2*t/d if d else 1., t/union if union else 1.)

def main():
    print("="*94)
    print("CASR-NET BLOCK 06 — THREE-ROW JOURNAL QUALITATIVE PILOT (NO INFERENCE)")
    print("="*94)
    csvfile = SRC / "archived_reproduction_comparison.csv"
    manifestfile = SRC / "reproduction_manifest.json"
    assert csvfile.is_file() and manifestfile.is_file(), "Run Block 05 successfully first."
    with csvfile.open(newline="", encoding="utf-8") as f:
        recorded = {(r["case_id"], int(r["slice_idx"])): r for r in csv.DictReader(f)}
    baseline_manifest = json.loads(manifestfile.read_text(encoding="utf-8"))
    assert baseline_manifest.get("status") == "ALL_MATCH", "Block 05 did not verify all 5 archived examples."
    assert baseline_manifest.get("checkpoint_sha256") == CHECKPOINT_SHA256, "Checkpoint manifest mismatch."
    assert baseline_manifest.get("matches") == 5, "Expected five matching archived cases."

    # Images remain the genuine archived checkpoint outputs; 1 unique case per row.
    rows=[]
    for case, z in SELECTED:
        rec = recorded.get((case,z))
        assert rec and rec["status"] == "MATCH", f"Unverified selected case: {case}, {z}"
        file = SRC / f"{case}_slice_{z:04d}.npz"
        assert file.is_file(), f"Missing saved prediction arrays: {file}"
        with np.load(file, allow_pickle=False) as f:
            ct = f["ct_image"].copy()
            gt = f["ground_truth"].astype(bool)
            pred = f["prediction"].astype(bool)
        assert ct.shape == gt.shape == pred.shape == (256,256), f"Array shape mismatch: {case}"
        d, j = metrics(pred,gt)
        assert abs(d-float(rec["recovered_dice"])) < 1e-7
        assert abs(j-float(rec["recovered_iou"])) < 1e-7
        assert int(pred.sum()) == int(rec["recovered_pred_pixels"])
        rows.append((case,z,ct,gt,pred,d,j))
        print(f"VERIFIED: {case} slice={z} Dice={d:.6f}, IoU={j:.6f}, pred pixels={int(pred.sum())}")

    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(nrows=3,ncols=5,figsize=(16.3,9.6),
                          gridspec_kw={"wspace":0.028,"hspace":0.12})
    headers=("CT Image","Reference Infection","CASR-Net Prediction",
             "Boundary Overlay","Segmentation Errors")
    for j,title in enumerate(headers):
        axes[0,j].set_title(title,fontweight="bold",fontsize=12.0,pad=12)

    manifest_rows=[]
    for i,(case,z,ct,gt,pred,d,jac) in enumerate(rows):
        tp=gt & pred; fp=pred & (~gt); fn=gt & (~pred)
        err=np.zeros((*gt.shape,3),dtype=np.float32)
        err[tp]=(0.95,0.95,0.95)
        err[fp]=(0.93,0.16,0.15)
        err[fn]=(0.18,0.42,0.95)

        for ax in axes[i]:
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values(): spine.set_visible(False)

        axes[i,0].imshow(ct,cmap="gray",vmin=0,vmax=1)
        axes[i,1].imshow(gt,cmap="gray",vmin=0,vmax=1)
        axes[i,2].imshow(pred,cmap="gray",vmin=0,vmax=1)
        axes[i,3].imshow(ct,cmap="gray",vmin=0,vmax=1)
        if gt.any(): axes[i,3].contour(gt.astype(float),levels=[0.5],colors=["lime"],
                                      linewidths=1.0)
        if pred.any(): axes[i,3].contour(pred.astype(float),levels=[0.5],colors=["red"],
                                        linewidths=1.0)
        axes[i,4].imshow(err,interpolation="nearest")
        tag=chr(ord("A")+i)
        axes[i,0].text(0.03,0.965,f"{tag}  {case} · slice {z}",
                       transform=axes[i,0].transAxes,ha="left",va="top",
                       fontsize=9.0,fontweight="bold",color="white",
                       bbox={"boxstyle":"round,pad=0.22","facecolor":"black",
                             "alpha":0.72,"edgecolor":"none"})
        axes[i,2].text(0.5,0.035,f"Dice = {d:.3f}",
                       transform=axes[i,2].transAxes,ha="center",va="bottom",
                       fontsize=11.0,fontweight="bold",color="white",
                       bbox={"facecolor":"black","alpha":0.76,"edgecolor":"none",
                             "boxstyle":"round,pad=0.24"})
        manifest_rows.append({"row":tag,"case_id":case,"slice_idx":z,
                              "dice":d,"iou":jac,"gt_pixels":int(gt.sum()),
                              "pred_pixels":int(pred.sum()),
                              "npz_source":str(SRC / f"{case}_slice_{z:04d}.npz")})

    fig.subplots_adjust(left=0.018,right=0.985,top=0.943,bottom=0.087,
                        wspace=0.028,hspace=0.12)
    fig.text(0.5,0.043,
             "Boundary overlay: reference = green, prediction = red.    "
             "Error map: TP = white, FP = red, FN = blue.",
             ha="center",va="center",fontsize=10.5)
    DST.mkdir(parents=True,exist_ok=True)
    png=DST/"CASRNet_Qualitative_Three_Cases_Pilot.png"
    pdf=DST/"CASRNet_Qualitative_Three_Cases_Pilot.pdf"
    fig.savefig(png,dpi=300,facecolor="white")
    fig.savefig(pdf,facecolor="white")
    plt.close(fig)
    caption=("Qualitative pulmonary infection segmentation in three distinct held-out "
             "CT cases using the recovered CASR-Net student. Columns show the chest CT image, "
             "reference infection mask, student prediction, reference/prediction boundaries "
             "(green/red), and pixel-wise segmentation errors (true positive: white; "
             "false positive: red; false negative: blue). Dice is reported for each slice. "
             "The recovered visualization checkpoint is not identified as the final "
             "2.5-D student used for the primary quantitative results.")
    (DST/"Figure_Caption.txt").write_text(caption+"\n",encoding="utf-8")
    record={"created_utc":datetime.now(timezone.utc).isoformat(),
            "purpose":"3-row presentation of five-exemplar checkpoint recovery",
            "figure_is_final_manuscript_result":False,
            "selection":"Predeclared examples from three distinct held-out cases; not a test-set best/median/worst claim",
            "selected":manifest_rows,
            "original_five_examples_preserved":True,
            "checkpoint_sha256":CHECKPOINT_SHA256,
            "caption":caption,"png":str(png),"pdf":str(pdf)}
    (DST/"Figure_Manifest.json").write_text(json.dumps(record,indent=2)+"\n",
                                             encoding="utf-8")
    print("\nTHREE-ROW FIGURE:",png)
    print("PRINT PDF:",pdf)
    print("CAPTION:",DST/"Figure_Caption.txt")
    print("MANIFEST:",DST/"Figure_Manifest.json")
    print("ORIGINAL FIVE-ROW FIGURE AND PREDICTIONS: UNCHANGED")
    print("="*94)

if __name__ == "__main__":
    main()
