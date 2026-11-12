#!/bin/bash
# Map SLURM_ARRAY_TASK_ID -> one arm, shared by every GPU phase of experiment 5.
#
# Sharding by arm is safe for all four phases because each writes its own
# results/<arm>/tN_*.json — there is no shared file for tasks to clobber, and
# aggregate.py reads whatever is present. Sourcing this sets ARM_FLAG, which the caller
# splices into the python invocation ahead of "$@" so an explicit --arms still wins.
#
# Keep this list in the same order as common.ARM_ORDER.
ARMS=(mgnify pfam_fullchain pfam_legacy pfam_domain pf00107)
ARM_FLAG=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]:-}"
  if [ -z "${ARM}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} is out of range for ${#ARMS[@]} arms" >&2
    exit 2
  fi
  echo "array task ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}"
  ARM_FLAG=(--arms "${ARM}")
fi
