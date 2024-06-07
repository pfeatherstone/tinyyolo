
mkdir weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights -P weights
wget https://pjreddie.com/media/files/yolov3.weights -P weights
wget https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-spp.weights -P weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights -P weights
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -P weights
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
echo "import torch" >> script.py
echo "torch.save(torch.load('yolov7.pt', map_location='cpu')['model'].float().state_dict(), '../weights/yolov7.pt')" >> script.py
python3 script.py
cd ..
rm -rf yolov7
