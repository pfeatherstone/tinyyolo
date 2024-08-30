mkdir weights
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

# Yolov7 massaging
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
echo "import torch" >> script.py
echo "torch.save(torch.load('yolov7.pt', map_location='cpu')['model'].float().state_dict(), '../weights/yolov7.pt')" >> script.py
python3 script.py
cd ..
rm -rf yolov7

# Yolov6 massaging
git clone https://github.com/meituan/YOLOv6
cd YOLOv6

rm -f model.txt
rm -f script.py
echo "import sys" >> script.py
echo "import torch" >> script.py
echo "d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)" >> script.py
echo "f = open('model.txt', 'w')" >> script.py
echo "f.write(str(d['model']))" >> script.py
echo "f.close()" >> script.py
echo "torch.save(d['model'].state_dict(), f'../weights/{sys.argv[1]}')" >> script.py

wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt
python3 script.py yolov6n.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
python3 script.py yolov6s.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt
python3 script.py yolov6m.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt
python3 script.py yolov6l.pt

cd ..
rm -rf YOLOv6