#!/bin/bash

WORKSPACE_PATH="$1"
IMAGES_PATH="$2"
MAX_IMAGE_SIZE=1600
IS_MULTPLE_MODELS=1

function automatic_reconstructor() {
    echo "Automatic Reconstructor"
    # {low, medium, high, extreme}
    colmap automatic_reconstructor \
        --workspace_path "$WORKSPACE_PATH" \
        --image_path "$IMAGES_PATH" \
        --quality medium \
        --camera_model PINHOLE \
        --single_camera 1
    save_statistics
}

function save_statistics() {
    echo "Save statistics about reconstructions"
    for d in "$WORKSPACE_PATH/sparse/"*; do
        colmap model_analyzer \
            --path "$d" \
            > "$d/stats.txt" 2>&1

        colmap model_converter \
            --input_path "$d" \
            --output_path "$d/model.nvm" \
            --output_type NVM
    done
}

if [ ! -d "$WORKSPACE_PATH" ]; then
    echo "Invalid workspace folder"
    exit 1
fi

if [ ! -d "$IMAGES_PATH" ]; then
    echo "Invalid image folder"
    exit 1
fi

automatic_reconstructor
