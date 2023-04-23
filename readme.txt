******************************************************************************

# Live multi video real time ANPR Computer vision app

******************************************************************************

# Goal ----------------------------------------

Goal was to make ANPR system to make it fast, accurate and to be able to work 24/7

# Project description  ------------------------

ANPR is based on Python,
YOLOv5s trained detection
PaddleOCR text detection and recognition
roi DNN super resolution
perspective transform
duplicate check
regional country errors fix
blacklist mail notification,
saving cropped license plates to web server,
inserting ocr text to mysql db,
with php bootstrap and datatables based GUI for live view, search and daily, monthly and ip cam stats!

This version is without regional country errors fix, DB and web GUI with stats and made for offline video files and/or ip cams
IP cam version has different approach and is made for possible network failure and has automatic ip cam reconnection even if network hw goes down

******************************************************************************

# Python install instructions -----------------

Download and install:
https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe
https://aka.ms/vs/17/release/vc_redist.x64.exe

# Make folder structure -----------------------

ANPR
-dnn_superres
-lp
-mail
-video
-weight
-yolov5

# Python requirements -------------------------

From ANPR folder run cmd:
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall opencv-python
pip install opencv-contrib-python

if Nvidia GPU
instead:
- paddlepaddle
- torch torchvision torchaudio

install:
- cuda_11.6.1 and cudnn 8.5 drivers from Nvidia
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
https://docs.nvidia.com/deeplearning/cudnn/install-guide/#install-windows

https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/windows-pip_en.html
- pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
- pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

download https://github.com/ultralytics/yolov5
and copy all content in ANPR/yolov5 folder 

download https://github.com/fannymonori/TF-ESPCN/blob/master/export/ESPCN_x4.pb
and copy to ANPR/dnn_superres folder

******************************************************************************

# YOLOv5 detection weight ---------------------

Train your Yolov5s weight for licence plates, rename it to best.pt and copy to ANPR/weight folder
Yolov5s detection is trained for european lp, possible it will work for other regions also

******************************************************************************

# Demo video files ----------------------------

Download video files from https://drive.google.com/drive/folders/1RWdhSj6TntGe8Qoinp63TzAK3H7p91gX?usp=sharing
and copy to ANPR/video folder

******************************************************************************

# Blacklist mail notification -----------------
In blacklist.py change mail account settings

smtp_server = 'smtp.domain.com' # mail smtp
sender = 'user@domain.com'  # Enter your address
receiver = 'user@domain.com'  # Enter receiver address
password = '' # mail password

in blacklist.txt insert few licence plates which would be detected from videos in folder ANPR/video, each in new row without dashes, commas or space e.g.

******************************************************************************

# Other settings-------------------------------

FPS optimization in anpr.py is set to True for frame skiping
Change to False if have GPU for testing

******************************************************************************

# Run code ------------------------------------

After all is set run anpr.py
all detected license plates would be saved in ANPR/lp folder with structure ANPR/lp/year/month/day/hour/minute/timestampe-video file name-lp.jpg

******************************************************************************

Made and tested on Windows 10/11 i9, 64 GB RAM with nVidia GPU (2080 and 3070TI) PC and on Lenovo laptop wih i7 16 GB RAM without GPU

Minimum hardware requirements:
Windows 11
SSD 120 GB
i5/i7 processor
16 GB RAM
