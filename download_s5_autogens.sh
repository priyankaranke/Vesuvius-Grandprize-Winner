#!/bin/bash

USERNAME="registered_user"
PASSWORD="only"

BASE_URL="community-uploads/bruniss/scrolls/s5/autogens/"

echo "Downloading all segments from s5 autogens folder..."

# Download all layers recursively from the s5 autogens directory
# This will create the directory structure: ./train_scrolls/scroll5/{segment_id}/layers/
rclone copy ":http:/${BASE_URL}" "./train_scrolls/scroll5/" \
    --http-url "http://$USERNAME:$PASSWORD@dl.ash2txt.org/" \
    --progress \
    --multi-thread-streams=8 \
    --transfers=8 \
    --include "*/layers/**" \
    --create-empty-src-dirs

echo "All downloads complete." 