import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import cv2
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

# Load FaceNet model for embedding extraction
embedder = FaceNet()
facenet_model = embedder.model
detector = MTCNN()

def fancyDraw(img, bbox, l=30, t=3, rt=1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    cv2.rectangle(img, bbox, (255, 0, 255), rt)
    # Top Left x, y
    cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
    cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
    # Top Right x1, y
    cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
    # Bottom Left x, y1
    cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
    # Bottom Right x1, y1
    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

# Extract face embeddings from a given frame
def get_face_embeddings(frame):
    faces = detector.detect_faces(frame)
    embeddings = []
    boxes = []
    for face in faces:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face_pixels = frame[y1:y2, x1:x2]
        face_pixels = cv2.resize(face_pixels, (160, 160))
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = facenet_model.predict(samples)
        embeddings.append(yhat[0])
        boxes.append((x1, y1, x2, y2))
    return embeddings, boxes

# Capture video from the webcam and make predictions
def realtime_face_recognition(model, out_encoder):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        face_embeddings, boxes = get_face_embeddings(frame)
        
        for i, face_emb in enumerate(face_embeddings):
            samples = expand_dims(face_emb, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            predict_name = out_encoder.inverse_transform(yhat_class)[0]
            
            x1, y1, x2, y2 = boxes[i]
            bbox = (x1, y1, x2-x1, y2-y1)
            cv2.putText(frame, f'{predict_name} ({class_probability:.2f}%)', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            fancyDraw(frame, bbox)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    with open('svm_model/svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('svm_model/out_encoder.pkl', 'rb') as encoder_file:
        out_encoder = pickle.load(encoder_file)

    realtime_face_recognition(model, out_encoder)

if __name__ == "__main__":
    main()
    

