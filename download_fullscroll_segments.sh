#!/bin/bash

USERNAME="registered_user"
PASSWORD="only"

SCROLL_NUMBER=4
SCROLL_NAME="PHerc1667"
SEGMENT_IDS=("20240304161941")

# Read segment IDs from the file and download them
for segment_id in "${SEGMENT_IDS[@]}"; do
    echo "--- Downloading segment: $segment_id ---"
    
    # Create directory if it doesn't exist
    mkdir -p "./train_scrolls/${segment_id}/layers"
    
    # Download mask
    # rclone copy ":http:/full-scrolls/Scroll${SCROLL_NUMBER}/${SCROLL_NAME}.volpkg/paths/${segment_id}/${segment_id}_mask.png" "./train_scrolls/${segment_id}/" --http-url "http://$USERNAME:$PASSWORD@dl.ash2txt.org/" --progress --multi-thread-streams=8 --transfers=8
    
    # Download layers
    rclone copy ":http:/full-scrolls/Scroll${SCROLL_NUMBER}/${SCROLL_NAME}.volpkg/paths/${segment_id}/layers/" "./train_scrolls/${segment_id}/layers/" --http-url "http://$USERNAME:$PASSWORD@dl.ash2txt.org/" --progress --multi-thread-streams=8 --transfers=8
done

echo "All downloads complete." 
