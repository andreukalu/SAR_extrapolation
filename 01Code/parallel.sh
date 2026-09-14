#!/bin/bash

GRAPH="/home/usuaris/csl/andreu.salcedo/SAR/SAR_extrapolation/graph_merge_AS_remote.xml"
INPUT_DIR="/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/SLC"
OUTPUT_DIR="/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/PROCESSED"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over input files (e.g., .zip, .SAFE, .dim, or .tif)
for file in "$INPUT_DIR"/*.zip; do
  # Check if file exists
  [ -e "$file" ] || continue
  
  # Extract filename without extension
  base=$(basename "$file" .zip)
  
  echo "Processing $base..."
  
  # Run SNAP GPT
  /opt/esa-snap-v9/bin/gpt "$GRAPH" \
    -Pinput_file="$file" \
    -Poutput_file="$OUTPUT_DIR/${base}_processed.dim" \
    -f BEAM-DIMAP \
    -c 16G -q 8
done