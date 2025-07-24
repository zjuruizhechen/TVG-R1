from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yeliudev/VideoMind-Dataset",
    repo_type="dataset",
    local_dir="/opt/tiger/video-r1/data",
    allow_patterns=[
        "didemo/*",
        "tacos/*",
        "queryd/*",
        "hirest/*",
        "internvid_vtime/*",
    ]
)

