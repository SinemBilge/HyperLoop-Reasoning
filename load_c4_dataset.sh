#!/bin/bash

#Install Git LFS if not already installed


git lfs install

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4

# Download specific files or folders (replace "en/*" with specific patterns if needed)
FILES_ARG="--files=5"
for arg in "$@"; do
    if [[ $arg == --files=* ]]; then
        FILES_ARG=$arg
        break
    fi
done

python ../load_c4_dataset.py $FILES_ARG

echo "Download completed"