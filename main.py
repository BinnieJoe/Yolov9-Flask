from flask import Flask, render_template, request, url_for, Response
from werkzeug.utils import secure_filename
import cv2, os, torch
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# 이미지인식

# 경로 설정
UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    
    image_file = request.files['file']
    
    if image_file.filename == '':
        return "No selected file", 400
    
    if not allowed_file(image_file.filename):
        return "File type not allowed", 400
    
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    img = cv2.imread(image_path)
    model = YOLO("yolov9c.pt")
    results = model.predict(img)

    # 예측 결과 처리 및 시각화
    for result in results:
        for box in result.boxes:
            # 좌표 값을 추출하여 정수로 변환
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            class_id = box.cls[0].item()
            label = result.names[class_id]
            
            # 바운딩 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 저장
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
    cv2.imwrite(result_image_path, img)
    
    return render_template('image.html', result_image_url=url_for('static', filename=f'img/result_{filename}'))

# 웹캠인식

model = YOLO("yolov9c.pt")

def yolo_detect(img, model):
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    return boxes, scores, classes

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('카메라 연결 실패')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)

        res_boxes, res_scores, res_classes = yolo_detect(img_tensor, model)

        for box, score, cls in zip(res_boxes, res_scores, res_classes):
            x1, y1, x2, y2 = box.astype(int)
            label = f'{model.names[int(cls)]} {score:.2f}'
            colors = np.random.uniform(0, 255, size=(len(model.names), 3))
            color = colors[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam')
def cam():
    return render_template("cam.html")

if __name__ == '__main__':
    app.run(debug=True)
