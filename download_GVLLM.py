from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="WHB139426/Grounded-VideoLLM",
    repo_type="dataset",
    local_dir="/opt/tiger/video-r1/data/GroundedVLLM",
    allow_patterns=[
        "activitynet/*",
        "qvhighlights/*",
    ]
)

