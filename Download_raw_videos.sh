python download_GVLLM.py &
python download.py
wait

cd ./data/GroundedVLLM/Grounded-VideoQA
unzip chunk_1.zip
cd ..

cd activitynet
unzip -o chunk_1.zip &
unzip -o chunk_2.zip &
unzip -o chunk_3.zip &
unzip -o chunk_4.zip
wait
find . -maxdepth 1 -name 'chunk_[1-4].zip' -delete
cd ..

cd qvhighlights
unzip -o chunk_1.zip &
unzip -o chunk_2.zip &
unzip -o chunk_3.zip &
unzip -o chunk_4.zip &
unzip -o chunk_5.zip
wait
find . -maxdepth 1 -name 'chunk_[1-5].zip' -delete
cd ..

cd ../..


cd ./data

cd didemo
find . -maxdepth 1 -name 'videos_crop_3fps_480_noaudio.tar.gz' -delete
cat videos.tar.gz.* > videos.tar.gz
tar -xzf videos.tar.gz
find . -maxdepth 1 -name 'videos.tar.gz.0[0-1]' -delete
find . -maxdepth 1 -name 'videos.tar.gz' -delete
cd ..

cd queryd
find . -maxdepth 1 -name 'videos_crop_3fps_480_noaudio.tar.gz' -delete
cat videos.tar.gz.* > videos.tar.gz
tar -xzf videos.tar.gz
find . -maxdepth 1 -name 'videos.tar.gz.0[0-1]' -delete
find . -maxdepth 1 -name 'videos.tar.gz' -delete
cd ..

cd tacos
find . -maxdepth 1 -name 'videos_crop_3fps_480_noaudio.tar.gz' -delete
tar -xzf videos.tar.gz
find . -maxdepth 1 -name 'videos.tar.gz' -delete
cd ..

cd internvid_vtime
find . -maxdepth 1 -name 'videos_crop_3fps_480_noaudio.tar.gz.0[0-2]' -delete
cat videos.tar.gz.* > videos.tar.gz
tar -xzf videos.tar.gz
find . -maxdepth 1 -name 'videos.tar.gz.0[0-9]' -delete
find . -maxdepth 1 -name 'videos.tar.gz' -delete
cd ..

cd hirest
find . -maxdepth 1 -name 'videos_crop_3fps_480_noaudio.tar.gz.0[0-1]' -delete
cat videos.tar.gz.* > videos.tar.gz
tar -xzf videos.tar.gz
find . -maxdepth 1 -name 'videos.tar.gz.0[0-4]' -delete
find . -maxdepth 1 -name 'videos.tar.gz' -delete
cd ..


pip install nncore==0.4.5
sudo apt install -y libgl1-mesa-glx
pip install termplotlib==0.3.9
pip install pysrt==1.1.2
pip install opencv-python
