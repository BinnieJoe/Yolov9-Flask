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
