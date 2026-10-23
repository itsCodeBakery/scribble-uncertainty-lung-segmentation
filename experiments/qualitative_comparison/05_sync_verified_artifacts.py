#!/usr/bin/env python3
"""Block 07 — Safely sync verified CASR-Net Kaggle figure outputs to GitHub.

Scope: the approved three-row pilot, archived five-example reproduction,
single-slice diagnostic, and read-only audit JSON. Never stages source CT
volumes, original datasets, teacher/checkpoint weights, or credentials.

Requires: Kaggle Secret GITHUB_TOKEN (fine-grained, repository Contents: write).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "casrnet-qualitative-comparison-2026"
BASE = Path("experiments/qualitative_comparison")
DIRS = [
    BASE / "outputs/casrnet_recovered_pilot",
    BASE / "outputs/casrnet_recovered_five_examples",
    BASE / "outputs/casrnet_three_row_pilot",
]
AUDITS = [
    BASE / "audits/block00_environment.json",
    BASE / "audits/block03_casrnet_input_compatibility.json",
]
INCLUDE = {".npz", ".png", ".pdf", ".json", ".csv", ".txt"}
MAX_BYTES = 24 * 1024 * 1024  # Regular git; never upload large raw data
SYNC_MANIFEST = BASE / "sync/artifact_checksums.csv"
SYNC_SUMMARY = BASE / "sync/last_verified_sync.json"
REQUIRED = [
    BASE / "outputs/casrnet_recovered_five_examples/reproduction_manifest.json",
    BASE / "outputs/casrnet_recovered_five_examples/archived_reproduction_comparison.csv",
    BASE / "outputs/casrnet_recovered_five_examples/CASRNet_Recovered_Five_Examples.png",
    BASE / "outputs/casrnet_recovered_five_examples/CASRNet_Recovered_Five_Examples.pdf",
    BASE / "outputs/casrnet_three_row_pilot/Figure_Manifest.json",
    BASE / "outputs/casrnet_three_row_pilot/Figure_Caption.txt",
    BASE / "outputs/casrnet_three_row_pilot/CASRNet_Qualitative_Three_Cases_Pilot.png",
    BASE / "outputs/casrnet_three_row_pilot/CASRNet_Qualitative_Three_Cases_Pilot.pdf",
]

def git(*args, env=None, check=True):
    command = ["git", "-C", str(ROOT), *args]
    p = subprocess.run(command, text=True, capture_output=True, env=env)
    if check and p.returncode:
        raise RuntimeError(f"Git command failed ({args[0]}): {p.stderr[:700]}")
    return p.stdout.strip(), p.returncode

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def token_from_kaggle():
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        v = None
    return v.strip() if isinstance(v, str) and v.strip() else None

def verify_source():
    head, _ = git("branch", "--show-current")
    assert head == BRANCH, f"Wrong Git branch: {head}; expected {BRANCH}"
    status, _ = git("diff", "--cached", "--name-only")
    assert not status, "Other files are already staged. Review them before syncing."

    for item in REQUIRED:
        assert (ROOT / item).is_file(), f"Missing required artifact: {item}"

    five = json.loads((ROOT / REQUIRED[0]).read_text())
    assert five.get("status") == "ALL_MATCH" and five.get("matches") == 5, (
        "Five-example reproduction did not match archived results."
    )
    three = json.loads((ROOT / (BASE/"outputs/casrnet_three_row_pilot/Figure_Manifest.json")).read_text())
    assert len(three.get("selected", [])) == 3, "Three-row manifest is incomplete."
    unique_cases = {row["case_id"] for row in three["selected"]}
    assert len(unique_cases) == 3, "Three-row figure must contain distinct cases."

    arrays = list((ROOT / DIRS[1]).glob("*.npz"))
    assert len(arrays) == 5, f"Expected 5 saved .npz prediction arrays; found {len(arrays)}"
    print("✓ Five recovered examples verified: 5/5 MATCH")
    print("✓ Three-row figure manifest verified: three distinct cases")
    return five, three

def collect_files():
    files=[]
    for rel in DIRS:
        root = ROOT / rel
        if not root.is_dir():
            print("Not present (skipped):", rel)
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.is_symlink() or p.suffix.lower() not in INCLUDE:
                continue
            if p.stat().st_size > MAX_BYTES:
                print(f"SKIPPED (>24 MiB): {p.relative_to(ROOT)}")
                continue
            files.append(p)
    for rel in AUDITS:
        p = ROOT / rel
        if p.is_file() and not p.is_symlink():
            files.append(p)
    return sorted(set(files))

def main():
    print("=" * 94)
    print("CASR-NET BLOCK 07 — VERIFIED FIGURE AND EXPERIMENT SYNCHRONIZATION")
    print("=" * 94)
    verify_source()
    files = collect_files()
    assert files, "No experiment artifacts to upload."

    print("\nFILES IDENTIFIED FOR GITHUB:")
    for p in files:
        print(f"  {str(p.relative_to(ROOT)):<105} {p.stat().st_size/1024:8.1f} KB")
    print("TOTAL:", len(files), "files;", round(sum(x.stat().st_size for x in files)/1e6, 2), "MB")

    # Use the original output files, never a compressed chat screenshot.
    token = token_from_kaggle()
    if not token:
        print("\nGITHUB_TOKEN Kaggle Secret is missing or inaccessible.")
        print("Kaggle notebook: Add-ons > Secrets > Add a secret called GITHUB_TOKEN.")
        print("Use a fine-grained token with Contents: Read and write permission")
        print("for only itsCodeBakery/scribble-uncertainty-lung-segmentation.")
        print("Enable the secret for this notebook and re-run Block 07.")
        print("No Git commit or push attempted; source outputs remain intact.")
        return

    # Public fetch before any local commit, to avoid a non-fast-forward push.
    git("pull", "--ff-only", "origin", BRANCH)

    SYNC_MANIFEST_PATH = ROOT / SYNC_MANIFEST
    SYNC_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SYNC_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path","bytes","sha256"])
        for p in files:
            writer.writerow([str(p.relative_to(ROOT)),p.stat().st_size,sha256(p)])

    meta = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "branch": BRANCH,
        "reproduction": "5/5 archived recovered checkpoints predictions match",
        "figure_layout": "three rows / five columns / three distinct test cases",
        "source": "original Kaggle high-resolution outputs",
        "files_count": len(files),
        "bytes_total": sum(p.stat().st_size for p in files),
        "large_weights_or_datasets_uploaded": False,
        "note": "Recovered qualitative model is NOT the identified final 2.5D checkpoint."
    }
    (ROOT/SYNC_SUMMARY).write_text(json.dumps(meta,indent=2)+"\n", encoding="utf-8")
    staged = [str(p.relative_to(ROOT)) for p in files] + [str(SYNC_MANIFEST),str(SYNC_SUMMARY)]
    for rel in staged:
        git("add", "--", rel)

    changed, _ = git("diff", "--cached", "--name-only")
    if not changed:
        print("\nAll identified artifacts were already committed. Checking remote.")
    else:
        print("\nSTAGED CHANGES:")
        print(changed)
        name, _ = git("config", "user.name", check=False)
        email, _ = git("config", "user.email", check=False)
        if not name: git("config", "user.name", "CASR-Net Research")
        if not email: git("config", "user.email", "casrnet-research@users.noreply.github.com")
        git("commit", "-m", "results: sync verified CASR-Net three-row and recovery artifacts")

    # GIT_ASKPASS reads the token only from the subprocess environment:
    # never place it in git config, notebook code, remote URLs, or commit.
    fd, askpass = tempfile.mkstemp(prefix="casrnet_askpass_", suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(
                '#!/bin/sh\n'
                'case "$1" in\n'
                '  *Username*) printf "%s\\n" "x-access-token";;\n'
                '  *Password*) printf "%s\\n" "$GITHUB_TOKEN";;\n'
                '  *) printf "\\n";;\n'
                'esac\n'
            )
        os.chmod(askpass, 0o700)
        env = os.environ.copy()
        env.update({"GITHUB_TOKEN": token, "GIT_ASKPASS": askpass,
                    "GIT_TERMINAL_PROMPT": "0"})
        cmd=["git","-C",str(ROOT),"push","origin",f"HEAD:refs/heads/{BRANCH}"]
        p=subprocess.run(cmd,capture_output=True,text=True,env=env)
        if p.returncode:
            detail=(p.stderr or "No further details").replace(token,"[REDACTED]")
            raise RuntimeError("GitHub push failed: "+detail[:1100])
        print("\n✓ GitHub push completed.")
    finally:
        try: os.unlink(askpass)
        except FileNotFoundError: pass

    # Verify GitHub remote SHA = local HEAD SHA without printing secrets.
    local, _ = git("rev-parse", "HEAD")
    remote, _ = git("ls-remote", "origin", f"refs/heads/{BRANCH}")
    remote_sha=remote.split()[0] if remote else ""
    print("Local Git commit :", local)
    print("Remote branch SHA :", remote_sha)
    assert local == remote_sha, "Remote Git commit SHA did not match local HEAD."
    print("✓ REMOTE VERIFIED — all selected files synchronized.")
    print("GitHub branch: https://github.com/itsCodeBakery/"
          "scribble-uncertainty-lung-segmentation/tree/"+BRANCH)
    print("Historical datasets, model weights and training files were not overwritten.")
    print("=" * 94)

if __name__ == "__main__":
    main()
