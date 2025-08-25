#!/usr/bin/env python3

# ================================================================
# Evaluation *looper* script
# - Runs each subject‑specific checkpoint
# - Prints overall means
# - Saves subject‑wise results to csv file
# Shortcut for generating a copy-paste-friendly folder list:
# find . -maxdepth 1 -type d -name '*LOSO*' -printf '"%P",\n'
# ================================================================

# argparse flag --csv_name
# name of the per-subject csv file

from pathlib import Path
import numpy as np
import torch
import pandas as pd
import event_classification as ec
import argparse

parser = argparse.ArgumentParser(
    description="Loop over checkpoints, evaluate, and export a subject-wise CSV."
)
parser.add_argument(
    "--csv_name",
    type=str,
    default="subject_wise_accuracy.csv",
    help="Filename for the per-subject CSV (it will be placed in ./output/)",
)
args = parser.parse_args()
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / args.csv_name

model_folders = [

# TODO
# Comma-delimited str folder names

]

# No Lookahead
COMMON_KWARGS = dict(
    eval_batch_size=64,
    eval_task="predict_action",
    transition_samples_only=False,
    buffer_range=200,
    mask_percentage=0.6,
    mask_type="poisson",
    stride=20,
    files_or_dirs=["../data/ROAM_EMG"],
    allow_relax=0,
    lookahead=0,
    weight_max_factor=1.0,
    likelihood_format="logits",
    samples_between_prediction=1,
    maj_vote_range="single",
    epn_eval=0,
    recog_threshold=0.5,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
)

"""
# With lookahead
COMMON_KWARGS = dict(
    eval_batch_size=64,
    eval_task="predict_action",
    transition_samples_only=False,
    buffer_range=200,
    mask_percentage=0.6,
    mask_type="poisson",
    stride=1,
    files_or_dirs=["../data/ROAM_EMG"],
    allow_relax=0,
    lookahead=50,
    weight_max_factor=1.0,
    likelihood_format="logits",
    samples_between_prediction=20,
    maj_vote_range="future",
    epn_eval=0,
    recog_threshold=0.5,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
)
"""

# ---------------------------------------
#                ACCUMULATORS
# ---------------------------------------
event_accs = []
raw_accs = []
subject_ids = []

# ------------------------------------------
#                MAIN LOOP
# ------------------------------------------
for folder in model_folders:
    ckpt = Path("model_checkpoints") / folder / "epoch_9.pth"

    # --- Determine the subject ID from the checkpoint’s args_dict ---
    ckpt_dict = torch.load(ckpt, map_location="cpu")
    args_dict = ckpt_dict["args_dict"]

    if "val_patient_ids" in args_dict:
        subj_id = args_dict["val_patient_ids"][0]
    else:
        raise Exception(
            "val_patient_ids not in saved args_dict. Verify checkpoint integrity"
        )

    print(f"→ evaluating {ckpt}  (subject {subj_id})")

    evt_acc, raw_acc = ec.main(
        saved_checkpoint_pth=str(ckpt),
        **COMMON_KWARGS,
    )

    # save results
    subject_ids.append(subj_id)
    event_accs.append(evt_acc)
    raw_accs.append(raw_acc)

# --------------------------------------------------
#              FINAL SUMMARY  (stdout)
# --------------------------------------------------
print("\n=========== FINAL SUMMARY ===========")
print(f"Models evaluated            : {len(model_folders)}")
print(f"Average Transition Accuracy : {np.mean(event_accs):.4f}")
print(f"Average Raw Accuracy        : {np.mean(raw_accs):.4f}")
print("=====================================")

# -----------------------------------------------
#              WRITE PER‑SUBJECT CSV
# -----------------------------------------------
df = pd.DataFrame(
    {
        "subject_id": subject_ids,
        "avg_raw_accuracy": raw_accs,
        "avg_transition_accuracy": event_accs,
    }
)

df.to_csv(csv_path, index=False)
print(f"Per-subject results written to '{csv_path}'.")
