import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
model_yaml_path = './cfg/models/rt-detr/rtdetr-l.yaml'
data_yaml_path = './dataset/data.yaml'
if __name__ == '__main__':
    model = RTDETR(model_yaml_path)
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=10,
                          batch=4,
                          workers=0,
                          project='runs/RT-DETR/train',
                          name='exp',
                          )