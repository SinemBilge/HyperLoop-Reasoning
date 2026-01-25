from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

# Download specific dataset files
file = hf_hub_download(
    repo_id="fira-aslan/ml-lab-wiki",
    filename="dataset.zip",
    repo_type="dataset",
    local_dir="./",
    token="",
    local_dir_use_symlinks=False
)

model_file = hf_hub_download(
    repo_id="fira-aslan/HyperLoopReasoning",
    filename="knit5.pth",
    token="",
    local_dir="./",
    local_dir_use_symlinks=False
)

local_path = snapshot_download(
    repo_id="fira-aslan/HyperLoopReasoning",
    allow_patterns="runs/*",  # Optional: specify folder pattern
    token="",
    local_dir="./",
    local_dir_use_symlinks=False
)

