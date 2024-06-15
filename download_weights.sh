
mkdir weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights -P weights
wget https://pjreddie.com/media/files/yolov3.weights -P weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights -P weights
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -P weights
wget https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-spp.weights -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5nu.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5su.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5mu.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5lu.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5xu.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt -P weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt -P weights

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
echo "import torch" >> script.py
echo "torch.save(torch.load('yolov7.pt', map_location='cpu')['model'].float().state_dict(), '../weights/yolov7.pt')" >> script.py
python3 script.py
cd ..
rm -rf yolov7
