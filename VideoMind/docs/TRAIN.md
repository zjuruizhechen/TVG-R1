# Training VideoMind

## 🛠️ Environment Setup

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 11.8 / CANN 8.0
- Python 3.11.0
- PyTorch 2.4.0 / Torch-NPU 2.4.0.post2
- [Transformers](https://github.com/huggingface/transformers) 4.45.2
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) 0.15.4
- [NNCore](https://github.com/yeliudev/nncore) 0.4.5

### Install the environment

1. Clone the repository from GitHub.

```shell
git clone git@github.com:yeliudev/VideoMind.git
cd VideoMind
```

2. Initialize conda environment.

```shell
conda create -n videomind python=3.11 -y
conda activate videomind
```

3. Install dependencies.

```shell
pip install -r requirements.txt
```

For NPU users, please modify [Line 18-25](https://github.com/yeliudev/VideoMind/blob/main/requirements.txt#L18:L25) of `requirements.txt`.

### Prepare base models

Download [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), then place them into the `model_zoo` folder.

```
VideoMind
└─ model_zoo
   ├─ Qwen2-VL-2B-Instruct
   └─ Qwen2-VL-7B-Instruct
```

## 📦 Dataset Preparation

The training data used for each role is listed as follows. All the data, including the raw videos, compressed videos, and annotations, could be downloaded on [Hugging Face](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset).

| Role | Datasets |
|-|-|
| `Grounder` | `qvhighlights`, `didemo`, `tacos`, `queryd`, `cosmo_cap`, `internvid_vtime`, `hirest_grounding`, `hirest_step` |
| `Verifier` | `verifying` |
| `Planner` | `planning` |

The codebase also supports more grounding datasets such as `ego_timeqa`, `ego4d_nlq`, `ego4d_naq`, `vid_morp`, `videoxum`, and `youcook2`, but they are not used to train the model in our paper. Different variants of these datasets are also included. See the individual dataset files [here](https://github.com/yeliudev/VideoMind/blob/main/videomind/dataset/sub_classes) for more details.

After downloading the required datasets, extract the `tar.gz` files and place them in the `data` folder. The processed files should be organized in the following structure (taking `charades_sta` as an example).

```
VideoMind
└─ data
   └─ charades_sta
      ├─ videos_3fps_480_noaudio
      ├─ durations.json
      ├─ charades_sta_train.txt
      └─ charades_sta_test.txt
```

## 🕹️ Start Training

Use the following commands to train VideoMind. We recommand using **NVIDIA A100 (80G) GPUs** or **Ascend 910B (65G) NPUs**. The default setting is to use 8 devices on a single node. You may modify `nproc_per_node`, `per_device_train_batch_size`, and `gradient_accumulation_steps` to keep the same global batch size (32) if you have different device configurations.

```shell
# Pretrain Grounder (2B / 7B)
bash scripts/pretrain/pretrain_grounder_2b.sh
bash scripts/pretrain/pretrain_grounder_7b.sh

# Pretrain Verifier (2B / 7B)
bash scripts/pretrain/pretrain_verifier_2b.sh
bash scripts/pretrain/pretrain_verifier_7b.sh

# Pretrain Planner (2B / 7B)
bash scripts/pretrain/pretrain_planner_2b.sh
bash scripts/pretrain/pretrain_planner_7b.sh

# Finetune Grounder on QVHighlights (2B / 7B)
bash scripts/finetune/finetune_qvhighlights_2b.sh
bash scripts/finetune/finetune_qvhighlights_7b.sh
```

The training logs and checkpoints will be saved in the `work_dirs` folder. After training all the roles, you may modify the checkpoint paths in [eval_auto_2b.sh](https://github.com/yeliudev/VideoMind/blob/main/scripts/evaluation/eval_auto_2b.sh) or [eval_auto_7b.sh](https://github.com/yeliudev/VideoMind/blob/main/scripts/evaluation/eval_auto_7b.sh) for evaluation.
