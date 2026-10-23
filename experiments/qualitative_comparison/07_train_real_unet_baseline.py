#!/usr/bin/env python3
"""Block 09: Train genuine U-Net baseline on frozen train/validation case splits.

Uses no external pretrained weights. Select checkpoint and operating threshold
ONLY on validation slices; use three preselected test cases for qualitative
comparison. This is a new model, not the historic unavailable U-Net checkpoint.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
from datetime import datetime, timezone

import cv2
import nibabel as nib
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"experiments/qualitative_comparison/outputs"
PREDS=BASE/"casrnet_recovered_five_examples"
OUT=BASE/"sota_unet_effb0_vs_casrnet"
DATA=ROOT/"data/splits"
CKPT=OUT/"unet_effb0_validation_best.pth.gz"
SELECTED=[("radiopaedia_7",32),("coronacases_009",120),("coronacases_001",125)]
TTA=[(),(-1,),(-2,),(-2,-1)]
SEED=42
EPOCHS=30
PATIENCE=9
BATCH=12

def read_rows(p):
    with p.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def hash_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda:f.read(4194304),b""):
            h.update(part)
    return h.hexdigest()

def prep_ct(x):
    x=np.clip(np.asarray(x,dtype=np.float32),-1000,400)
    return cv2.resize((x+1000)/1400,(256,256),
                      interpolation=cv2.INTER_LINEAR).astype(np.float32)

def prep_mask(y):
    return cv2.resize((np.asarray(y)>0).astype(np.uint8),(256,256),
                      interpolation=cv2.INTER_NEAREST).astype(np.uint8)

class SegDataset(Dataset):
    def __init__(self,records,index,augment=False):
        self.records=records
        self.index=index
        self.augment=augment
        self.loaded={}
    def __len__(self):
        return len(self.records)
    def vols(self,case):
        if case not in self.loaded:
            x=self.index[case]
            ps=[Path(x[k]) for k in ("ct_path","infection_mask_path","lung_mask_path")]
            for p in ps: assert p.is_file(), f"Missing NIfTI: {p}"
            self.loaded[case]=[nib.load(str(p)) for p in ps]
        return self.loaded[case]
    def __getitem__(self,i):
        r=self.records[i];z=int(r["slice_idx"]);case=r["case_id"]
        ct,infect,lung=self.vols(case)
        x=prep_ct(ct.dataobj[:,:,z])
        native=np.asarray(infect.dataobj[:,:,z])>0
        if int(native.sum())!=int(r["infection_pixels"]):
            raise ValueError(f"Original split mismatch: {case} {z}")
        y=prep_mask(native)
        lung=prep_mask(lung.dataobj[:,:,z])
        if self.augment:
            if random.random()<.5:
                x=x[:,::-1].copy();y=y[:,::-1].copy();lung=lung[:,::-1].copy()
            if random.random()<.5:
                x=x[::-1].copy();y=y[::-1].copy();lung=lung[::-1].copy()
        return (torch.from_numpy(np.ascontiguousarray(x[None])),
                torch.from_numpy(np.ascontiguousarray(y[None].astype(np.float32))),
                torch.from_numpy(np.ascontiguousarray(lung[None].astype(bool))))

def model_fn():
    return smp.Unet(encoder_name="efficientnet-b0",encoder_weights=None,
                    in_channels=1,classes=1,activation=None)

def dice(pred,gt):
    tp=int((pred & gt).sum())
    n=int(pred.sum())+int(gt.sum())
    return 2*tp/n if n else 1.0

def conf(pred,gt):
    return {"TP":int((pred & gt).sum()),
            "FP":int((pred & ~gt).sum()),
            "FN":int((~pred & gt).sum()),
            "TN":int((~pred & ~gt).sum())}

def prob_tta(model,x):
    views=[]
    with torch.inference_mode():
        for axes in TTA:
            xx=torch.flip(x,axes) if axes else x
            y=torch.sigmoid(model(xx))
            if axes:y=torch.flip(y,axes)
            views.append(y.float().cpu().numpy())
    return np.mean(np.stack(views),axis=0)

def main():
    print("="*94)
    print("BLOCK 09 — TRAIN AND VISUALIZE A REAL U-NET BASELINE")
    print("="*94)
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/"configs/preprocessing_config.json").read_text())
    assert (cfg["image_size"],cfg["hu_clip_min"],cfg["hu_clip_max"])==(256,-1000,400)
    idx={r["case_id"]:r for r in read_rows(DATA/"dataset_index.csv")}
    tr=read_rows(DATA/"train_slices.csv")
    va=read_rows(DATA/"val_slices.csv")
    te=read_rows(DATA/"test_slices.csv")
    groups=[{r["case_id"] for r in part} for part in (tr,va,te)]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2]),"Case leakage between data splits"
    assert all((c,z) in {(r["case_id"],int(r["slice_idx"])) for r in te}
               for c,z in SELECTED)
    print("Frozen slices train/val/test:",len(tr),len(va),len(te))
    print("Distinct case counts:",[len(x) for x in groups])
    print("Training architecture: 2-D U-Net with EfficientNet-B0 encoder; random initialization")

    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    net=model_fn().to(device)
    print("Trainable parameters:",sum(p.numel() for p in net.parameters() if p.requires_grad))
    trainloader=DataLoader(SegDataset(tr,idx,augment=True),batch_size=BATCH,shuffle=True,
                           num_workers=2,pin_memory=device.type=="cuda")
    validloader=DataLoader(SegDataset(va,idx),batch_size=BATCH,shuffle=False,
                           num_workers=2,pin_memory=device.type=="cuda")
    existing=OUT/"comparison_manifest.json"
    done=existing.exists() and CKPT.exists() and json.loads(existing.read_text()).get("status")=="COMPLETED"
    history=[]
    if not done:
        loss_bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.],device=device))
        optimizer=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
        scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
        best=-1.0;epoch_best=0;bad=0
        for epoch in range(1,EPOCHS+1):
            net.train();losses=[]
            for x,y,_ in tqdm(trainloader,desc=f"Train {epoch}/{EPOCHS}",leave=False):
                x=x.to(device);y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                    logits=net(x)
                    p=torch.sigmoid(logits)
                    intersects=(p*y).sum((1,2,3))
                    total=p.sum((1,2,3))+y.sum((1,2,3))
                    loss=.5*loss_bce(logits,y)+.5*(1-(2*intersects+1)/(total+1)).mean()
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                losses.append(float(loss.item()))
            net.eval();TP=0;PP=0;GG=0
            with torch.inference_mode():
                for x,y,lung in tqdm(validloader,desc="Validate",leave=False):
                    x=x.to(device)
                    pred=(torch.sigmoid(net(x)).cpu()>=.5)&lung
                    gt=(y>=.5)
                    TP+=int((pred & gt).sum())
                    PP+=int(pred.sum())
                    GG+=int(gt.sum())
            vd=2*TP/max(1,PP+GG)
            history.append({"epoch":epoch,"train_loss":np.mean(losses),"val_micro_dice_t050":vd})
            with (OUT/"training_history.csv").open("w",newline="") as f:
                wr=csv.DictWriter(f,fieldnames=history[0].keys())
                wr.writeheader();wr.writerows(history)
            if vd>best+1e-6:
                best=vd;epoch_best=epoch;bad=0
                pack={"model_state_dict":{k:v.detach().cpu() for k,v in net.state_dict().items()},
                      "epoch":epoch,"model_name":"unet_effb0_2d_independent",
                      "best_validation_dice_t050":vd,
                      "train_csv_sha256":hash_file(DATA/"train_slices.csv"),
                      "val_csv_sha256":hash_file(DATA/"val_slices.csv")}
                with gzip.open(CKPT,"wb",compresslevel=5) as f:
                    torch.save(pack,f)
            else:bad+=1
            print(f"Epoch {epoch:02d}: train_loss={np.mean(losses):.5f}; "
                  f"val_dice={vd:.5f}; best={best:.5f} (epoch {epoch_best})")
            if epoch>=10 and bad>=PATIENCE:
                print("Early stopping on validation.");break
    else:
        print("Existing COMPLETED independent U-Net run detected; reusing checkpoint.")

    with gzip.open(CKPT,"rb") as f:
        pack=torch.load(f,map_location="cpu",weights_only=True)
    assert pack["train_csv_sha256"]==hash_file(DATA/"train_slices.csv")
    assert pack["val_csv_sha256"]==hash_file(DATA/"val_slices.csv")
    net.load_state_dict(pack["model_state_dict"],strict=True);net.eval()
    print("Best U-Net validation epoch:",pack["epoch"])

    # Select fixed threshold ONLY on validation slices; hold test images aside.
    probs=[];gts=[];lungs=[]
    with torch.inference_mode():
        for x,y,lung in tqdm(validloader,desc="Choose validation threshold",leave=False):
            p=prob_tta(net,x.to(device))
            probs.append(p[:,0])
            gts.append(y[:,0].numpy()>0.5)
            lungs.append(lung[:,0].numpy()>0.5)
    probs=np.concatenate(probs);gts=np.concatenate(gts);lungs=np.concatenate(lungs)
    search=[]
    for threshold in np.round(np.arange(.05,.951,.05),3):
        pred=(probs>=threshold)&lungs
        search.append({"threshold":float(threshold),"validation_micro_dice":dice(pred,gts)})
    chosen=max(search,key=lambda r:r["validation_micro_dice"])
    with (OUT/"validation_threshold_selection.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=search[0].keys())
        wr.writeheader();wr.writerows(search)
    print(f"Validation-selected threshold={chosen['threshold']:.2f}; "
          f"validation Dice={chosen['validation_micro_dice']:.5f}")

    previous=BASE/"sota_pilot_unetpp_vs_casrnet"
    unetpp_exists=all(
        (previous/f"{c}_slice_{z:04d}_unetpp_25d_pilot.npz").is_file()
        for c,z in SELECTED)
    heads=["CT Image","Ground Truth","U-Net (Eff-B0)"]+(
        ["UNet++ (2.5-D)"] if unetpp_exists else [])+[
        "CASR-Net (Ours)","Boundary Overlay","TP / FP / FN / TN"]
    fig,ax=plt.subplots(3,len(heads),figsize=(3.2*len(heads),9.6))
    for j,h in enumerate(heads):ax[0,j].set_title(h,fontsize=11,fontweight="bold")
    all_counts=[]
    for i,(case,z) in enumerate(tqdm(SELECTED,desc="Create matched test figure")):
        with np.load(PREDS/f"{case}_slice_{z:04d}.npz",allow_pickle=False) as a:
            img=a["ct_image"].copy()
            gt=a["ground_truth"].astype(bool)
            lung=a["lung_mask"].astype(bool)
            casr=a["prediction"].astype(bool)
        x=torch.from_numpy(img[None,None]).float().to(device)
        p=prob_tta(net,x)[0,0]
        upred=(p>=chosen["threshold"])&lung
        np.savez_compressed(OUT/f"{case}_slice_{z:04d}_unet.npz",
                            probability=p,prediction=upred.astype(np.uint8))
        up=None
        if unetpp_exists:
            with np.load(previous/f"{case}_slice_{z:04d}_unetpp_25d_pilot.npz",
                         allow_pickle=False) as a:
                assert np.array_equal(a["reference"].astype(bool),gt)
                up=a["baseline_prediction"].astype(bool)
        panels=[img,gt,upred]+([up] if up is not None else [])+[casr,img]
        for j,panel in enumerate(panels):
            ax[i,j].imshow(panel,cmap="gray",vmin=0,vmax=1)
        for mask,color in [(gt,"lime"),(upred,"yellow")]+(
            [(up,"cyan")] if up is not None else [])+[(casr,"red")]:
            if mask.any():
                ax[i,-2].contour(mask.astype(np.float32),[.5],
                                 colors=[color],linewidths=1)
        err=np.zeros((*gt.shape,3),dtype=np.float32)
        err[gt&casr]=(.97,.97,.97)
        err[~gt&casr]=(.93,.16,.15)
        err[gt&~casr]=(.2,.45,.95)
        ax[i,-1].imshow(err,interpolation="nearest")
        for j,pred in [(2,upred)]+([(3,up)] if up is not None else [])+[(len(heads)-3,casr)]:
            ax[i,j].text(.5,.035,f"Dice = {dice(pred,gt):.3f}",
                         transform=ax[i,j].transAxes,ha="center",
                         fontsize=10,fontweight="bold",color="white",
                         bbox={"facecolor":"black","alpha":.76,"edgecolor":"none"})
        c=conf(casr,gt)
        ax[i,-1].text(.5,.035,f"TP {c['TP']:,} | FP {c['FP']:,} | FN {c['FN']:,}",
                      transform=ax[i,-1].transAxes,fontsize=8,color="white",
                      fontweight="bold",ha="center",
                      bbox={"facecolor":"black","alpha":.76,"edgecolor":"none"})
        ax[i,0].text(.02,.97,f"{chr(65+i)} {case} / {z}",
                     transform=ax[i,0].transAxes,va="top",fontsize=8,
                     fontweight="bold",color="white",
                     bbox={"facecolor":"black","alpha":.76,"edgecolor":"none"})
        for a in ax[i]:
            a.set_xticks([]);a.set_yticks([])
            for spine in a.spines.values():spine.set_visible(False)
        all_counts.append({"case_id":case,"slice_idx":z,
                           "unet":conf(upred,gt),
                           "unetpp":conf(up,gt) if up is not None else None,
                           "casr":conf(casr,gt)})
        print(f"{case} {z}: new U-Net Dice={dice(upred,gt):.4f}; "
              f"CASR-Net Dice={dice(casr,gt):.4f}")
    fig.subplots_adjust(left=.007,right=.99,top=.943,bottom=.08,
                        wspace=.02,hspace=.105)
    fig.text(.5,.04,
             "Boundaries: GT = green | U-Net = yellow"+(
               " | UNet++ = cyan" if unetpp_exists else "")+
             " | CASR-Net = red    Errors: TP white, FP red, FN blue, TN black",
             ha="center",fontsize=9.5)
    png=OUT/"Real_UNet_vs_CASRNet_ThreeRows.png"
    pdf=OUT/"Real_UNet_vs_CASRNet_ThreeRows.pdf"
    fig.savefig(png,dpi=300,facecolor="white")
    fig.savefig(pdf,facecolor="white")
    plt.close(fig)
    result={
        "status":"COMPLETED",
        "method":"new independently trained U-Net with EfficientNet-B0 encoder (2D)",
        "not_historical_unet_baseline":True,
        "training":"original train_slices.csv case split, no pretrained weights",
        "validation":"original val_slices.csv case split",
        "threshold_selection":"validation-only grid 0.05 through 0.95",
        "selected_validation_threshold":chosen,
        "best_epoch":pack["epoch"],
        "train_csv_sha256":hash_file(DATA/"train_slices.csv"),
        "val_csv_sha256":hash_file(DATA/"val_slices.csv"),
        "test_csv_sha256":hash_file(DATA/"test_slices.csv"),
        "checkpoint_sha256":hash_file(CKPT),
        "model_checkpoint":str(CKPT),
        "unetpp_existing_pilot_included":unetpp_exists,
        "three_rows":all_counts,
        "casr_provenance":"original recovered qualitative example predictions; not final 2.5D quantitative checkpoint",
        "disclaimer":"Pilot qualitative comparison; no final SOTA claims or test-set tuning",
        "created_utc":datetime.now(timezone.utc).isoformat(),
    }
    (OUT/"comparison_manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print("\n✓ Figure PNG:",png)
    print("✓ Figure PDF:",pdf)
    print("✓ Best model compressed checkpoint:",CKPT)
    print("✓ Validation threshold search:",OUT/"validation_threshold_selection.csv")
    print("✓ Training history:",OUT/"training_history.csv")
    print("✓ Model provenance:",OUT/"comparison_manifest.json")
    print("No training on validation/test. No threshold selection on test.")
    print("="*94)

if __name__=="__main__":main()
