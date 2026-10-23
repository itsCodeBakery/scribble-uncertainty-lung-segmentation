#!/usr/bin/env python3
"""Block 08: Generate first ACTUAL matched qualitative baseline comparison.

UNet++ 2.5-D fully supervised checkpoint vs previously reproduced CASR-Net
recovered student. Three frozen held-out CT cases; NO training or manual masks.

IMPORTANT: This is a PILOT, not a validated SOTA benchmark. The UNet++ 2.5-D
checkpoint is a separate training configuration from the published nine-method
table. The baseline threshold 0.50 is preliminary, not a validation-optimized
choice. CASR-Net uses its recorded recovered-example threshold 0.035.
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
import requests
import torch
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments/qualitative_comparison/outputs/casrnet_recovered_five_examples"
OUT = ROOT / "experiments/qualitative_comparison/outputs/sota_pilot_unetpp_vs_casrnet"
BASELINE_CHECKPOINT = Path("/kaggle/working/casrnet_preservation_audit/unetplusplus_25d_clean_posloss_best_model.pth")
BASELINE_URL = ("https://raw.githubusercontent.com/itsCodeBakery/"
                "scribble-uncertainty-lung-segmentation/main/"
                "outputs/checkpoints/unetplusplus_efficientnetb0_full_supervision_25d_clean_posloss/"
                "best_model.pth")
BASELINE_SIZE = 26638717
BASELINE_GIT_BLOB = "fc14ef80bb12e63d2f0aa27d19973462b78abfef"
PILOT_BASELINE_THRESHOLD = 0.5
CASR_THRESHOLD = 0.035
SELECTED = (("radiopaedia_7", 32), ("coronacases_009", 120),
            ("coronacases_001", 125))
FLIPS = ((), (-1,), (-2,), (-2, -1))

def hash_file(p):
    digest = hashlib.sha256()
    blob = hashlib.sha1()
    n = p.stat().st_size
    blob.update(f"blob {n}\0".encode())
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(chunk)
            blob.update(chunk)
    return digest.hexdigest(), blob.hexdigest()

def obtain_checkpoint():
    BASELINE_CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    if not BASELINE_CHECKPOINT.is_file():
        tmp = BASELINE_CHECKPOINT.with_suffix(".part")
        try:
            with requests.get(BASELINE_URL,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                size=int(response.headers.get("content-length") or 0)
                with tmp.open("wb") as f:
                    for chunk in tqdm(
                        response.iter_content(1024*1024),
                        total=(size+1048575)//1048576 if size else None,
                        desc="Download UNet++ preserved checkpoint",unit="MiB"):
                        if chunk: f.write(chunk)
            tmp.replace(BASELINE_CHECKPOINT)
        finally:
            if tmp.exists(): tmp.unlink()
    n = BASELINE_CHECKPOINT.stat().st_size
    sha256, blob = hash_file(BASELINE_CHECKPOINT)
    if n != BASELINE_SIZE or blob != BASELINE_GIT_BLOB:
        raise RuntimeError("Baseline checkpoint does not match verified GitHub blob. Stop.")
    print(f"✓ Baseline checkpoint integrity verified: {n} bytes, SHA256={sha256}")
    return sha256

def load_baseline(device):
    ckpt = torch.load(BASELINE_CHECKPOINT,map_location="cpu",weights_only=True)
    if not isinstance(ckpt,dict): raise RuntimeError("Unexpected checkpoint container")
    if "model_state_dict" in ckpt: state=ckpt["model_state_dict"]
    elif "state_dict" in ckpt: state=ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"],dict): state=ckpt["model"]
    else: state=ckpt
    if any(x.startswith("module.") for x in state):
        state={x.removeprefix("module."):v for x,v in state.items()}
    stem = state.get("encoder._conv_stem.weight")
    if stem is None:
        raise RuntimeError("EfficientNet-B0 stem missing: checkpoint needs architecture review.")
    channels=int(stem.shape[1])
    assert channels in (1,3),f"Unexpected baseline input channels: {channels}"
    model=smp.UnetPlusPlus(encoder_name="efficientnet-b0",
                          encoder_weights=None,
                          in_channels=channels,classes=1,activation=None)
    try:
        model.load_state_dict(state,strict=True)
    except RuntimeError as e:
        raise RuntimeError("Cannot load ALL baseline weights strictly; no partial loading.") from e
    model=model.to(device).eval()
    print(f"✓ Loaded fully supervised UNet++ checkpoint (input channels={channels})")
    print("  Checkpoint metadata:",{k:str(ckpt[k])[:80] for k in
                                  ("epoch","model_name","best_val_dice","val_dice")
                                  if k in ckpt})
    return model,channels

def readcsv(p):
    with p.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def performance(pred,gt):
    p=pred.astype(bool); y=gt.astype(bool)
    intersect=int((p&y).sum()); union=int((p|y).sum())
    count=int(p.sum())+int(y.sum())
    return (2*intersect/count if count else 1.,
            intersect/union if union else 1.)

def as_256(arr,lo,hi):
    arr=np.clip(np.asarray(arr,dtype=np.float32),lo,hi)
    arr=(arr-lo)/(hi-lo)
    return cv2.resize(arr,(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)

def tta_probability(model,x):
    probs=[]
    with torch.inference_mode():
        for dims in FLIPS:
            aug=torch.flip(x,dims) if dims else x
            p=torch.sigmoid(model(aug))
            if dims: p=torch.flip(p,dims)
            probs.append(p.float().cpu().numpy()[0,0])
    return np.mean(np.stack(probs),axis=0).astype(np.float32)

def main():
    print("="*100)
    print("CASR-NET BLOCK 08 — ACTUAL SAME-SLICE SOTA QUALITATIVE COMPARISON: UNet++ VS CASR-NET")
    print("="*100)
    provenance = obtain_checkpoint()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,channels=load_baseline(device)
    cfg=json.loads((ROOT/"configs/preprocessing_config.json").read_text())
    lo,hi=float(cfg["hu_clip_min"]),float(cfg["hu_clip_max"])
    assert (int(cfg["image_size"]),lo,hi)==(256,-1000.,400.)
    source={r["case_id"]:r for r in readcsv(ROOT/"data/splits/dataset_index.csv")}
    test={(r["case_id"],int(r["slice_idx"]))
          for r in readcsv(ROOT/"data/splits/test_slices.csv")}
    assert all(pair in test for pair in SELECTED), "Selection contains non-test slices."
    manifest_path=SRC/"reproduction_manifest.json"
    assert manifest_path.is_file()
    recovered=json.loads(manifest_path.read_text())
    assert recovered.get("status")=="ALL_MATCH","CASR source must be fully verified."

    OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,5,figsize=(16.4,9.5))
    titles=("CT Image","Reference Infection","UNet++ (2.5-D)","CASR-Net (Recovered)",
            "Boundary Comparison")
    for j,t in enumerate(titles):
        axes[0,j].set_title(t,fontsize=11.5,fontweight="bold",pad=10)
    rows=[]
    data_volumes={}
    for row_idx,(case,k) in enumerate(tqdm(SELECTED,desc="Matched baseline inference")):
        archive=SRC/f"{case}_slice_{k:04d}.npz"
        if not archive.is_file(): raise FileNotFoundError(f"Run Block 05 first: {archive}")
        with np.load(archive,allow_pickle=False) as z:
            image=z["ct_image"].copy()
            gt=z["ground_truth"].astype(bool)
            lung=z["lung_mask"].astype(bool)
            casr_pred=z["prediction"].astype(bool)

        if case not in data_volumes:
            meta=source[case]
            path=Path(meta["ct_path"])
            if not path.is_file(): raise FileNotFoundError(f"Missing original CT: {path}")
            data_volumes[case]=nib.load(str(path))
        vol=data_volumes[case]
        n=vol.shape[2]
        # True 2.5-D input: previous, current, next axial CT slices.
        center=as_256(vol.dataobj[:,:,k],lo,hi)
        if not np.allclose(center,image,atol=1e-6):
            raise RuntimeError(f"Reconstructed central CT differs from verified CASR image: {case}/{k}")
        prev=as_256(vol.dataobj[:,:,max(0,k-1)],lo,hi)
        nex=as_256(vol.dataobj[:,:,min(k+1,n-1)],lo,hi)
        inp=np.stack([prev,center,nex],axis=0) if channels==3 else center[None]
        x=torch.from_numpy(inp[None]).float().to(device)
        p=tta_probability(model,x)
        # Fixed provisional baseline threshold, not selected on these test slices.
        baseline=(p>=PILOT_BASELINE_THRESHOLD)&lung
        base_d,base_j=performance(baseline,gt)
        casr_d,casr_j=performance(casr_pred,gt)
        arr_name=f"{case}_slice_{k:04d}"
        np.savez_compressed(OUT/f"{arr_name}_unetpp_25d_pilot.npz",
                            baseline_probability=p,
                            baseline_prediction=baseline.astype(np.uint8),
                            reference=gt.astype(np.uint8))
        axes[row_idx,0].imshow(image,cmap="gray",vmin=0,vmax=1)
        axes[row_idx,1].imshow(gt,cmap="gray",vmin=0,vmax=1)
        axes[row_idx,2].imshow(baseline,cmap="gray",vmin=0,vmax=1)
        axes[row_idx,3].imshow(casr_pred,cmap="gray",vmin=0,vmax=1)
        axes[row_idx,4].imshow(image,cmap="gray",vmin=0,vmax=1)
        for mask,color in ((gt,"lime"),(baseline,"cyan"),(casr_pred,"red")):
            if mask.any():
                axes[row_idx,4].contour(mask.astype(np.float32),levels=[0.5],
                                        colors=[color],linewidths=1)
        axes[row_idx,0].text(0.03,0.96,f"{chr(65+row_idx)}  {case} · slice {k}",
                              transform=axes[row_idx,0].transAxes,color="white",va="top",
                              fontsize=8.5,fontweight="bold",
                              bbox={"facecolor":"black","alpha":0.72,"edgecolor":"none",
                                    "boxstyle":"round,pad=0.2"})
        for col,d in ((2,base_d),(3,casr_d)):
            axes[row_idx,col].text(0.5,0.035,f"Dice = {d:.3f}",
                                    transform=axes[row_idx,col].transAxes,
                                    fontsize=10.5,color="white",ha="center",va="bottom",
                                    fontweight="bold",
                                    bbox={"facecolor":"black","alpha":0.75,
                                          "boxstyle":"round,pad=0.22","edgecolor":"none"})
        for ax in axes[row_idx]:
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        rows.append({"case_id":case,"slice_idx":k,"unetpp_25d_pilot_dice":base_d,
                     "unetpp_25d_pilot_iou":base_j,
                     "casr_recovered_dice":casr_d,"casr_recovered_iou":casr_j,
                     "unetpp_foreground_pixels":int(baseline.sum()),
                     "casr_foreground_pixels":int(casr_pred.sum()),
                     "gt_foreground_pixels":int(gt.sum())})
        print(f"  {arr_name}: UNet++ Dice={base_d:.4f} vs CASR-Net Dice={casr_d:.4f}")

    fig.subplots_adjust(left=0.012,right=0.992,top=0.944,bottom=0.085,
                        wspace=0.025,hspace=0.11)
    fig.text(0.5,0.045,
             "Boundary comparison: reference = green  |  UNet++ = cyan  |  CASR-Net = red",
             fontsize=10.5,ha="center",va="center")
    png=OUT/"SOTA_Pilot_UNetPP_vs_CASRNet_ThreeRows.png"
    pdf=OUT/"SOTA_Pilot_UNetPP_vs_CASRNet_ThreeRows.pdf"
    fig.savefig(png,dpi=300,facecolor="white")
    fig.savefig(pdf,facecolor="white")
    plt.close(fig)

    metrics=OUT/"same_slice_metrics.csv"
    with metrics.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    report={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PILOT_NOT_FINAL_SOTA_COMPARISON",
        "baseline":"UNet++ EfficientNet-B0, fully supervised 2.5-D clean-posloss checkpoint",
        "baseline_checkpoint_sha256":provenance,
        "baseline_input":"previous, central, next adjacent axial CT slices" if channels==3 else "single CT slice",
        "baseline_tta":"4-view horizontal/vertical flip ensemble",
        "baseline_threshold":PILOT_BASELINE_THRESHOLD,
        "baseline_lung_restriction":True,
        "baseline_threshold_validation_optimized":False,
        "casr_checkpoint_name":"casrnet_student_recovered_from_teacher_tta",
        "casr_threshold":CASR_THRESHOLD,
        "casr_note":"verified archived recovered qualitative prediction; not original final 2.5-D model",
        "data":"andrewmvd/covid19-ct-scans; fixed original 4-case test split",
        "rows":rows,
        "limitations":"Preliminary matched-mask demonstration; two checkpoints differ in training/input protocols and threshold selection. Do not use as final SOTA table or evidence for numeric aggregate improvement."
    }
    (OUT/"comparison_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    caption=("Matched pulmonary infection segmentation examples on three held-out chest CT "
             "slices, using a saved fully supervised 2.5-D UNet++ EfficientNet-B0 "
             "checkpoint and a separately recovered CASR-Net student. Boundaries show "
             "the reference (green), UNet++ (cyan), and CASR-Net (red). Per-slice Dice "
             "is annotated. Pilot comparison: baseline threshold 0.5 is not "
             "validation-optimized, and the recovered CASR-Net checkpoint is not "
             "the identified final 2.5-D quantitative model.")
    (OUT/"Figure_Caption.txt").write_text(caption+"\n")
    print("\n✓ Real UNet++ predictions generated on the SAME three CT slices as CASR-Net")
    print("PNG:",png)
    print("PDF:",pdf)
    print("Per-slice metrics:",metrics)
    print("Manifest:",OUT/"comparison_manifest.json")
    print("Original model weights and previous CASR-Net files were not modified.")
    print("="*100)

if __name__=="__main__":
    main()
