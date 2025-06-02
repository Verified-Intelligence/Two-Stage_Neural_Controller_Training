#!/bin/bash

# Usage: ./run_verify.sh <name> <c1> <c2> <path_to_model>

set -e  # Exit on error

# Current directory
CURRENT_DIR=$(pwd)

# Go to script directory so relative paths work
cd "$(dirname "$0")"

# Parse arguments
NAME="$1"
C1="$2"
C2="$3"
MODEL_PATH="$4"

# Declare associative array for box_radius
declare -A RADIUS_MAP
RADIUS_MAP[van]="4.8 10.8"
RADIUS_MAP[double]="26.4 9.6"
RADIUS_MAP[path_tracking_bigtorque]="10 10"
RADIUS_MAP[path_tracking_smalltorque]="10 10"
RADIUS_MAP[pendulum_bigtorque]="20 100"
RADIUS_MAP[pendulum_smalltorque]="19.2 64.8"
RADIUS_MAP[cartpole]="3.6 2.4 12 12"
RADIUS_MAP[2dquadrotor]="12 13.2 12 19.2 20.4 88.8"

# Check if NAME is in the map
if [[ -z "${RADIUS_MAP[$NAME]}" ]]; then
  echo "Error: Unknown name '$NAME'. Add it to the RADIUS_MAP in the script."
  exit 1
fi

# Get radius list
RADIUS="${RADIUS_MAP[$NAME]}"

# Run the Python script
python generate_vnnlib.py \
  --name "$NAME" \
  --box_radius $RADIUS \
  --c1 "$C1" \
  --c2 "$C2"

FULL_PATH="$CURRENT_DIR/$MODEL_PATH"

# Now go to the verification directory
cd ../alphabetaCROWN/complete_verifier
# Run the verification script
echo "Running verification for $NAME with seed $SEED"
python abcrown.py \
  --config ../../verification/$NAME/$NAME.yaml \
  --load_model "$FULL_PATH" \
# Return to the original directory
cd ../../../