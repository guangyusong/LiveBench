from huggingface_hub import snapshot_download

# Download the model files
model_path = snapshot_download(
    repo_id="fla-hub/rwkv7-2.9B-world",
    local_dir="./models/rwkv/rwkv7-2.9B-world",
    local_dir_use_symlinks=False
)
print(f"Model downloaded to: {model_path}")
