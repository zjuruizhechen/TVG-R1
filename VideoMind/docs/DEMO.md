# Local Gradio Demo for VideoMind

## 🛠️ Environment Setup

Please refer to [EVAL.md](/docs/EVAL.md) for setting up the environment and preparing the checkpoints.

Example videos can be downloaded at [here](https://huggingface.co/spaces/yeliudev/VideoMind-2B/tree/main/data).

## 🕹️ Launch Demo

Run the following command to launch Gradio demo locally.

```shell
# Set Python path
export PYTHONPATH="./:$PYTHONPATH"

# Launch demo
python demo/app.py
```

If success, the terminal would display `Running on local URL: http://0.0.0.0:7860`. You may then visit the link via your browser.
