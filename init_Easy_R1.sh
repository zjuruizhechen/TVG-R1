cd EasyR1-main
pip install -e .

pip install decord
pip install trl
pip install gpustat

cd qwen-vl-utils
pip install -e .
cd ..

pip install vllm==0.8.2

cd ..

cp /opt/tiger/video-r1/EasyR1-main/utils.py ~/.local/lib/python3.11/site-packages/vllm/model_executor/models/util.py

cp /opt/tiger/video-r1/EasyR1-main/modeling_qwen2_5_vl.py ~/.local/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py