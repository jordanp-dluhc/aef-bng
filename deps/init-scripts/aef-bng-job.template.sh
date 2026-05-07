#!/bin/bash
set -e

# --- Init script for DAB job clusters ---
# Installs uv and the aef-bng wheel from the bundle's synced artifacts.

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# The DAB syncs the wheel to the workspace file path alongside this script.
# Find it relative to this script's location.
SCRIPT_DIR=$(dirname "$0")
WHEEL_PATH=$(find "$SCRIPT_DIR" -name "aef_bng-*.whl" 2>/dev/null | head -n 1)

# Fallback: check the bundle's files directory (one level up from init-scripts)
if [ -z "$WHEEL_PATH" ]; then
  WHEEL_PATH=$(find "$SCRIPT_DIR/../files" -name "aef_bng-*.whl" 2>/dev/null | head -n 1)
fi

# Fallback: search the workspace bundle root
if [ -z "$WHEEL_PATH" ]; then
  WHEEL_PATH=$(find /Workspace/Users -path "*/.bundle/aef-bng/*/files/*" -name "aef_bng-*.whl" 2>/dev/null | head -n 1)
fi

if [ -z "$WHEEL_PATH" ]; then
  echo "Error: aef_bng wheel not found. Ensure the bundle deployed correctly."
  exit 1
fi

echo "Installing wheel via uv: ${WHEEL_PATH}"
uv pip install "${WHEEL_PATH}"
echo "aef-bng installation complete."
