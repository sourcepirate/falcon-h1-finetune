#!/bin/sh
# Resolve the job config, then exec the trainer.
# GOGGLE_RUN_CONFIG_B64 is injected when the image sets needs_config;
# GOGGLE_CONFIG_PATH (or --config) points at the in-pod path.
set -eu

if [ -n "${GOGGLE_RUN_CONFIG_B64:-}" ]; then
  dest="${GOGGLE_CONFIG_PATH:-/config/job.yaml}"
  mkdir -p "$(dirname "$dest")"
  printf '%s' "$GOGGLE_RUN_CONFIG_B64" | base64 -d > "$dest"
  echo "wrote config to $dest"
fi

exec python -m cyberft.cli "$@"
