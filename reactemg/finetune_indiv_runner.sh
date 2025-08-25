#!/usr/bin/env bash
set -euo pipefail

#############################################
# List of validation patient IDs (order matters)
#############################################
patient_ids=(
  "s1"  "s2"  "s3"  "s4"  "s5"  "s6"  "s7"
  "s8"  "s9"  "s10" "s11" "s12" "s13" "s14"
  "s15" "s16" "s17" "s18" "s19" "s20" "s21"
  "s22" "s23" "s24" "s25" "s26" "s27" "s28"
)

# Stroke example
# patient_ids=(
#   "p1"  "p16"
# )

#############################################
# Matching list of fine-tuning checkpoints
# MUST be in the same order and length
#############################################
checkpoint_pths=(

# TODO
# Space-delimited str folder names

)

if [[ ${#patient_ids[@]} -ne ${#checkpoint_pths[@]} ]]; then
  echo "ERROR: patient_ids and checkpoint_pths must have the same length." >&2
  exit 1
fi

for idx in "${!patient_ids[@]}"; do
  patient_id="${patient_ids[$idx]}"
  saved_checkpoint_pth="model_checkpoints/${checkpoint_pths[$idx]}/epoch_4.pth"

  # Optional: make sure the checkpoint file exists
  if [[ ! -f "$saved_checkpoint_pth" ]]; then
    echo "WARNING: Checkpoint $saved_checkpoint_pth not found - skipping $patient_id." >&2
    continue
  fi

  exp_name="Finetuned_${patient_id}"

  # LoRA-compatible
  # To use LoRA, add --use_lora 1 to cmd
  cmd=(
    python3 main.py
      --offset 30
      --num_classes 3
      --task_selection 0 1 2
      --use_input_layernorm
      --share_pe
      --dataset_selection finetune  # IMPORTANT! Set to enable finetuning
      --window_size 600
      --val_patient_ids "${patient_id}"
      --exp_name "${exp_name}"
      --epn_subset_percentage 1.0
      --epochs 5
      --model_choice any2any
      --inner_window_size 600
      --saved_checkpoint_pth "${saved_checkpoint_pth}"
  )

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
done
