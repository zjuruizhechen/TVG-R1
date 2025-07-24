
cd EasyR1-main
pip install -e .

pip install hf_xet
pip install torch==2.6.0
pip install vllm==0.8.2
pip install torchvision==0.21.0
pip install transformers==4.51.3
pip install hf_xet
pip install gpustat

cd ..

mkdir -p VideoMind/data

python download_videomind.py

cd ./VideoMind/data/

cd nextqa
tar -xzvf videos.tar.gz
cd ..

cd activitynet
find . -maxdepth 1 -name 'videos.tar.gz.0[0-9]' -delete
find . -maxdepth 1 -name 'videos.tar.gz.1[0-1]' -delete
cat videos_3fps_480_noaudio.tar.gz.* > videos_3fps_480_noaudio.tar.gz
tar -xzvf videos_3fps_480_noaudio.tar.gz
find . -maxdepth 1 -name 'videos_3fps_480_noaudio.tar.gz' -delete
cd ..

cd qvhighlights
find . -maxdepth 1 -name 'videos.tar.gz.0[0-2]' -delete
cat videos_3fps_480_noaudio.tar.gz.* > videos_3fps_480_noaudio.tar.gz
tar -xzvf videos_3fps_480_noaudio.tar.gz
find . -maxdepth 1 -name 'videos_3fps_480_noaudio.tar.gz.0[0-2]' -delete
cd ..

cd cgbench
find . -maxdepth 1 -name 'videos.tar.gz.0[0-7]' -delete
cat videos_3fps_480_noaudio.tar.gz.* > videos_3fps_480_noaudio.tar.gz
tar -xzvf videos_3fps_480_noaudio.tar.gz
find . -maxdepth 1 -name 'videos_3fps_480_noaudio.tar.gz.0[0-2]' -delete
cd ..

cd charades_sta
find . -maxdepth 1 -name 'videos.tar.gz.0[0-2]' -delete
tar -xzvf videos_3fps_480_noaudio.tar.gz
cd ..
