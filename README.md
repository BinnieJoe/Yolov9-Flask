# Flask-Based Object Detection Using YOLOv9
This project involves custom training the YOLOv9 model to recognize specific objects and developing a system that visually displays the results through a Flask web application. The dataset consists of images taken from various angles and lighting conditions, with each image containing bounding boxes and class labels.

After training the model using YOLOv9, we utilized Flask to implement a feature where users can upload images, and the system performs object detection, displaying the resulting images on the webpage. The webpage visually presents the bounding boxes and labels of the recognized objects, while also indicating the processing time to highlight the model's efficiency.

## Working
```
git clone https://github.com/ultralytics/yolov5.git
```

```
pip install -r requirements.txt
```

```
python main.py
```

## For Datatset
Making myGlob.py
```
from glob import glob
train_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/train/images/*.jpg')
test_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/test/images/*.jpg')
valid_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/valid/images/*.jpg')
print(len(train_img_list), len(test_img_list), len(valid_img_list))
import yaml

if len(train_img_list) > 0 :    
    with open('D:/hdh2024/BCCD/yolov5/dataSet/train.txt','w') as f:
        f.write('\n'.join(train_img_list) + '\n')
    with open('D:/hdh2024/BCCD/yolov5/dataSet/test.txt','w') as f:
        f.write('\n'.join(test_img_list) + '\n')
    with open('D:/hdh2024/BCCD/yolov5/dataSet/val.txt','w') as f:
        f.write('\n'.join(valid_img_list) + '\n')
```
Modifying the data.yaml file
```
train: D:/hdh2024/BCCD/yolov5/dataSet/train/images
val: D:/hdh2024/BCCD/yolov5/dataSet/valid/images

nc: 3
names: ['Platelets', 'RBC', 'WBC']
```
