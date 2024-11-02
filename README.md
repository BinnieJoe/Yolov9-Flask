# Flask-Based Object Detection Using YOLOv9
This project involves custom training the YOLOv9 model to recognize specific objects and developing a system that visually displays the results through a Flask web application. The dataset consists of images taken from various angles and lighting conditions, with each image containing bounding boxes and class labels.

After training the model using YOLOv9, we utilized Flask to implement a feature where users can upload images, and the system performs object detection, displaying the resulting images on the webpage. The webpage visually presents the bounding boxes and labels of the recognized objects, while also indicating the processing time to highlight the model's efficiency.

## Skills
- **Python**: Programming skills in Python for developing real-time object detection systems.
- **PyTorch**: Experience utilizing PyTorch for training deep learning models.
- **Roboflow**: Ability to use Roboflow for dataset management and preprocessing.
- **Flask**: Experience using the Flask framework to visualize real-time object detection results in web applications.
- **HTML**: Experience with HTML used for building web page layouts through Flask.

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
- Making myGlob.py
```
from glob import glob
train_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/train/images/*.jpg')
test_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/test/images/*.jpg')
valid_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/valid/images/*.jpg')
print(len(train_img_list), len(test_img_list), len(valid_img_list))
import yaml

if len(train_img_list) > 0 :    
    with open(' c:/joe/bin/yolov5/dataSet/train.txt','w') as f:
        f.write('\n'.join(train_img_list) + '\n')
    with open(' c:/joe/bin/yolov5/dataSet/test.txt','w') as f:
        f.write('\n'.join(test_img_list) + '\n')
    with open(' c:/joe/bin/yolov5/dataSet/val.txt','w') as f:
        f.write('\n'.join(valid_img_list) + '\n')
```
- Modifying the data.yaml file
```
train: c:/joe/bin/yolov5/dataSet/train/images
val: c:/joe/bin/yolov5/dataSet/valid/images

nc: 4
names: ['Car', 'Person', 'Traffic Light', 'Bicycle']
```
