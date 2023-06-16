import cv2
import threading
from flask import Flask, render_template, Response
import pyttsx3
from playsound import playsound
import os

app = Flask(__name__)
# URL video streaming
url = 'http://192.168.1.12:8080/video'

# Buat objek VideoCapture menggunakan URL
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

stop_program = False
classId_Result = []
classId_Result_lock = threading.Lock()

def process_image():
    global classId_Result, stop_program
    while not stop_program:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        with classId_Result_lock:
            classId_Result = classIds # Salin nilai classIds ke classId_Result
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)

        ret, frame = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def process_sound():
    global classId_Result, stop_program  # Menggunakan shared variable classIds dan stop_program

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Kecepatan bicara, misalnya 150 kata per menit
    engine.setProperty('volume', 1.0)  # Volume suara, dari 0.0 hingga 1.0

    while not stop_program:
        with classId_Result_lock:
            current_classId_Result = classId_Result  # Salin nilai classId_Result ke current_classId_Result

        with open('coco.names', 'rt') as f:
            labels = f.readlines()
            if len(current_classId_Result) > 0:
                label_first = current_classId_Result[0]
                label_number = label_first - 1
                label_index = label_number  # Indeks label yang ingin diakses

                if label_index < len(labels):
                    label = labels[label_index].strip()

                    engine.say(label)
                    engine.runAndWait()
                else:
                    print("Invalid label index.")


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Membuat thread untuk pemrosesan gambar dan pemutaran suara
image_thread = threading.Thread(target=process_image)
sound_thread = threading.Thread(target=process_sound)

# Menjalankan thread secara bersamaan
image_thread.start()
sound_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

    stop_program = True  # Mengubah nilai stop_program menjadi True
    image_thread.join()  # Menunggu thread image_thread selesai
    sound_thread.join()  # Menunggu thread sound_thread selesai
    cap.release()  # Melepas sumber video
    cv2.destroyAllWindows()  # Menutup jendela OpenCV
