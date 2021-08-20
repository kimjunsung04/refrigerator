import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
from PIL import ImageFont, ImageDraw, Image

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def train(name):
    data_path = "faces/" + name + "/"
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if images is None:
            continue
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    return model


def trains():
    data_path = "faces/"
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]

    models = {}
    for model in model_dirs:
        print("model :" + model)
        result = train(model)
        if result is None:
            continue
        print("model2 :" + model)
        models[model] = result

    return models


def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y : y + h, x : x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


def run(models):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
        try:
            min_score = 999
            min_score_name = ""

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            for key, model in models.items():
                result = model.predict(face)
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key

            if min_score < 500:
                confidence = int(100 * (1 - (min_score) / 300))
                display_string = str(confidence) + "% " + min_score_name
            cv2.putText(
                image,
                display_string,
                (100, 120),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (250, 120, 255),
                2,
            )
            if confidence > 80:
                cv2.putText(
                    image,
                    "Unlocked : " + min_score_name,
                    (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Face Cropper", image)
            else:
                cv2.putText(
                    image,
                    "Locked",
                    (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Face Cropper", image)
        except:
            cv2.putText(
                image,
                "Face Not Found",
                (250, 450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Face Cropper", image)
            pass
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    models = trains()
    run(models)
