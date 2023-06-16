import cv2
import threading
import pyttsx3

# Shared variable untuk menyimpan nilai classIds
classId_Result = []
classId_Result_lock = threading.Lock()  # Lock untuk sinkronisasi akses ke classId_Result

# Flag untuk menghentikan program
stop_program = False

def process_image():
    global classId_Result, stop_program  # Menggunakan shared variable classIds dan stop_program

    while not stop_program:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        print(classIds)

        with classId_Result_lock:
            classId_Result = classIds

        print("File Class ID Proces_Image =", classId_Result)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)
                cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)

        cv2.imshow("Output", img)
        key = cv2.waitKey(1)
        if key == 27:
            stop_program = True  # Mengubah flag stop_program menjadi True

def process_sound():
    global classId_Result, stop_program  # Menggunakan shared variable classIds dan stop_program

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Kecepatan bicara, misalnya 150 kata per menit
    engine.setProperty('volume', 1.0)  # Volume suara, dari 0.0 hingga 1.0

    while not stop_program:
        with classId_Result_lock:
            current_classId_Result = classId_Result

        with open('coco.names', 'rt') as f:
            print("File ClassId Process Sound =", current_classId_Result)
            labels = f.readlines()
            if len(current_classId_Result) > 0:
                label_first = current_classId_Result[0]
                label_number = label_first - 1
                label_index = label_number  # Indeks label yang ingin diakses
                print("Label Index =", label_index)
                print("ClassIds =", label_index)

                if label_index < len(labels):
                    label = labels[label_index].strip()
                    print("Label:", label)

                    engine.say(label)
                    engine.runAndWait()
                else:
                    print("Invalid label index.")



# Inisialisasi kamera, model, dan variabel lainnya
thres = 0.5
url = 'http://192.168.1.12:8080/video'
cap = cv2.VideoCapture(url)
cap.set(3, 640)
cap.set(3, 480)

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

# Menunggu thread selesai
image_thread.join()
sound_thread.join()

# Setelah selesai, tutup jendela OpenCV
cv2.destroyAllWindows()
