#!/usr/bin/env python3
"""Block 05 — Reproduce all five archived recovered CASR-Net qualitative examples.

A diagnostic validation of the recovered qualitative checkpoint only.
Do not interpret these as samples from the original final 2.5-D checkpoint.
Does not optimize thresholds or train any model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "qualitative_comparison" / "outputs" / "casrnet_recovered_five_examples"
CHECKPOINT = Path("/kaggle/working/casrnet_preservation_audit/recovered_student_best_model.pth")
CHECKPOINT_SHA256 = "97a6166d3b222187b90312929588196bf6797240b659d37eca93f8f49a06a020"
ARCHIVE = ROOT / "outputs" / "reports" / "CASRNet_Qualitative_Recovered_5x5_metadata.csv"
SPLIT = ROOT / "data" / "splits" / "test_slices.csv"
INDEX = ROOT / "data" / "splits" / "dataset_index.csv"
CFG = ROOT / "configs" / "preprocessing_config.json"
THRESHOLD = 0.035
FLIPS = ((), (-1,), (-2,), (-2,-1))

def readcsv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def performance(pred, label):
    a=pred.astype(bool); b=label.astype(bool)
    t=int(np.logical_and(a,b).sum())
    d=int(a.sum())+int(b.sum())
    u=int(np.logical_or(a,b).sum())
    return (2*t/d if d else 1.0, t/u if u else 1.0)

def main():
    print("="*96)
    print("CASR-NET BLOCK 05 — FIVE-EXAMPLE RECOVERED-CHECKPOINT REPRODUCTION")
    print("="*96)
    assert CHECKPOINT.exists(), "Recovered checkpoint not found; run Block 01."
    got_sha=sha256_file(CHECKPOINT)
    assert got_sha == CHECKPOINT_SHA256, "Recovered checkpoint SHA256 mismatch."
    print("CHECKPOINT VERIFIED:", got_sha)
    for path in (ARCHIVE, SPLIT, INDEX, CFG):
        assert path.is_file(), f"Missing repository metadata: {path}"

    archive=readcsv(ARCHIVE)
    assert len(archive)==5, f"Expected 5 archived examples; got {len(archive)}."
    test_keys={(x["case_id"],int(x["slice_idx"])) for x in readcsv(SPLIT)}
    index={x["case_id"]:x for x in readcsv(INDEX)}
    conf=json.loads(CFG.read_text())
    w=int(conf["image_size"]); lo=float(conf["hu_clip_min"]); hi=float(conf["hu_clip_max"])
    assert (w,lo,hi)==(256,-1000.,400.), "Historical preprocessing differs: stop and inspect."

    ck=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    model=smp.UnetPlusPlus(encoder_name="efficientnet-b0",encoder_weights=None,
                          in_channels=3,classes=1,activation=None)
    model.load_state_dict(ck["model_state_dict"],strict=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device).eval()
    print("CHECKPOINT MODEL:",ck.get("model_name"),"epoch:",ck.get("epoch"),"device:",device)
    print("ARCHIVED TEST EXAMPLES:",len(archive),"| Fixed TTA threshold:",THRESHOLD)
    print("TTA views: original, horizontal, vertical, horizontal+vertical")

    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    panels=[]
    volumes={}

    for record in tqdm(archive,desc="Archived example reproduction"):
        stem=Path(record["image_path"]).stem
        case,part=stem.split("_slice_")
        slice_idx=int(part)
        assert (case,slice_idx) in test_keys, f"Archived example NOT in frozen test split: {stem}"
        if case not in volumes:
            meta=index[case]
            def vol(key):
                p=Path(meta[key]); assert p.is_file(), f"Missing {p}"
                return nib.load(str(p))
            volumes[case]=(vol("ct_path"),vol("infection_mask_path"),vol("lung_mask_path"))
        ctvol,gtvol,lungvol=volumes[case]
        ct=np.asarray(ctvol.dataobj[:,:,slice_idx],dtype=np.float32)
        gt0=np.asarray(gtvol.dataobj[:,:,slice_idx])>0
        lung0=np.asarray(lungvol.dataobj[:,:,slice_idx])>0
        img=cv2.resize((np.clip(ct,lo,hi)-lo)/(hi-lo),(w,w),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32)
        gt=cv2.resize(gt0.astype(np.uint8),(w,w),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
        lung=cv2.resize(lung0.astype(np.uint8),(w,w),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        expected_gt=int(record["gt_pixels"])
        assert int(gt.sum())==expected_gt, (
            f"GT preprocessing mismatch for {stem}: got {int(gt.sum())}, expected {expected_gt}")

        x=torch.from_numpy(np.repeat(img[None],3,axis=0)[None]).float().to(device)
        probabilities=[]
        with torch.inference_mode():
            for dims in FLIPS:
                flipped=torch.flip(x,dims) if dims else x
                logits=model(flipped)
                p=torch.sigmoid(logits)
                if dims: p=torch.flip(p,dims)
                probabilities.append(p.float().cpu().numpy()[0,0])
        probability=np.stack(probabilities).mean(axis=0).astype(np.float32)
        prediction=(probability>=THRESHOLD)&lung
        dice,iou=performance(prediction,gt)
        ad=float(record["dice_internal_selection_only"])
        ai=float(record["iou_internal_selection_only"])
        ap=int(record["pred_pixels"])
        exact_pixels=int(prediction.sum())==ap
        close_metrics=abs(dice-ad)<1e-6 and abs(iou-ai)<1e-6
        status="MATCH" if (exact_pixels and close_metrics) else "DIFF"
        rows.append(dict(case_id=case,slice_idx=slice_idx,
                         recovered_dice=round(dice,10),archived_dice=round(ad,10),
                         recovered_iou=round(iou,10),archived_iou=round(ai,10),
                         recovered_pred_pixels=int(prediction.sum()),
                         archived_pred_pixels=ap,gt_pixels=int(gt.sum()),status=status,
                         checkpoint_sha256=got_sha))
        print(f"  {status:5s}  {stem:35s} "
              f"Dice={dice:.6f}/{ad:.6f} IoU={iou:.6f}/{ai:.6f} "
              f"PredPixels={int(prediction.sum())}/{ap}")

        # Save genuine probability/mask arrays, never hand-edited segmentations.
        np.savez_compressed(OUTPUT/f"{stem}.npz",
                            ct_image=img,ground_truth=gt.astype(np.uint8),
                            lung_mask=lung.astype(np.uint8),
                            student_probability=probability,
                            prediction=prediction.astype(np.uint8))

        tp=prediction&gt
        fp=prediction&~gt
        fn=~prediction&gt
        errors=np.zeros((w,w,3),dtype=np.float32)
        errors[tp]=(0.95,0.95,0.95)
        errors[fp]=(0.90,0.12,0.12)
        errors[fn]=(0.12,0.42,0.96)
        panels.append((stem,img,gt,prediction,probability,errors,dice))

    fig,axes=plt.subplots(len(panels),5,figsize=(17,3.5*len(panels)),
                          layout="constrained")
    titles=("CT image","Reference infection","Student prediction","Boundary comparison","TP / FP / FN error")
    for ax,title in zip(axes[0],titles):
        ax.set_title(title,fontsize=12,fontweight="bold")
    for i,(stem,img,gt,pred,prob,errors,d) in enumerate(panels):
        axes[i,0].imshow(img,cmap="gray",vmin=0,vmax=1)
        axes[i,1].imshow(gt,cmap="gray",vmin=0,vmax=1)
        axes[i,2].imshow(pred,cmap="gray",vmin=0,vmax=1)
        axes[i,3].imshow(img,cmap="gray",vmin=0,vmax=1)
        if gt.any(): axes[i,3].contour(gt.astype(float),levels=[0.5],colors=["lime"],linewidths=1)
        if pred.any(): axes[i,3].contour(pred.astype(float),levels=[0.5],colors=["red"],linewidths=1)
        axes[i,4].imshow(errors)
        for ax in axes[i]:
            ax.set_xticks([]);ax.set_yticks([])
        axes[i,0].set_ylabel(f"{stem}\nDice={d:.3f}",fontsize=9,fontweight="bold")
    fig.suptitle("CASR-Net: Reproduced Qualitative Results Using a Recovered Student Checkpoint",
                 fontsize=15,fontweight="bold")
    png=OUTPUT/"CASRNet_Recovered_Five_Examples.png"
    pdf=OUTPUT/"CASRNet_Recovered_Five_Examples.pdf"
    fig.savefig(png,dpi=260,bbox_inches="tight")
    fig.savefig(pdf,bbox_inches="tight")
    plt.close(fig)

    csvpath=OUTPUT/"archived_reproduction_comparison.csv"
    with csvpath.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    report={"created_utc":datetime.now(timezone.utc).isoformat(),
            "status":"ALL_MATCH" if all(r["status"]=="MATCH" for r in rows) else "DIFFERENCES_PRESENT",
            "matches":sum(r["status"]=="MATCH" for r in rows),
            "total":len(rows),
            "checkpoint_model_name":ck.get("model_name"),
            "checkpoint_epoch":ck.get("epoch"),
            "checkpoint_sha256":got_sha,
            "inference":"2D grayscale repeated across 3 channels, HU [-1000,400] normalized to [0,1], 4-view flip TTA; threshold 0.035, lung restriction",
            "study_scope":"Five archived recovered-qualitative examples; NOT final CASR-Net 0.8046 test score",
            "output_files":[str(p) for p in [csvpath,png,pdf]]}
    (OUTPUT/"reproduction_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print("\nREPRODUCTION STATUS:",report["status"],
          f"({report['matches']}/{report['total']} matching)")
    print("Saved CSV:",csvpath)
    print("Saved PNG:",png)
    print("Saved PDF:",pdf)
    print("Saved original probability/prediction arrays as five .npz files")
    print("Saved manifest:",OUTPUT/"reproduction_manifest.json")
    print("No training, hyperparameter selection, or baseline-model analysis.")
    print("="*96)

if __name__=="__main__":
    main()
