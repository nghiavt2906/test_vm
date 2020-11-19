import cv2, os, joblib, requests, json
import numpy as np
from time import time
from npsocket import SocketNumpyArray
from numpy import asarray, expand_dims
from PIL import Image
from fast_mtcnn import FastMTCNN
from sklearn.preprocessing import Normalizer, LabelEncoder
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from sklearn.svm import SVC

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.6,
    keep_all=True,
    device='cpu'
)

embedding_model = VGGFace(
    model='resnet50', 
    include_top=False, 
    input_shape=(224, 224, 3), 
    pooling='avg'
)

full_labels = []

def get_embedding(embedding_model, face):
    pixels = face.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    yhat = embedding_model.predict(samples)

    return yhat[0]

def get_models(embedding_model, isTrain):
    path = 'models/vgg_models'
    svc_filename, encoder_filename = 'svc_model.mdl', 'encoder_model.mdl'

    encoder_path = os.path.join(path, encoder_filename)
    svc_path = os.path.join(path, svc_filename)

    print('Loading the model from dir...')
    encoder = joblib.load(encoder_path)
    svc_model = joblib.load(svc_path)
    return svc_model, encoder

def main():
    isTrain = ""
    while isTrain != 'y' and isTrain != 'n':
        isTrain = input("Train the model or not (y/n)? ")

    svc_model, encoder = get_models(embedding_model, isTrain)
    print('Completed')

    start = time()
    count = 0
    fps = 0

    sock_receiver = SocketNumpyArray()
    sock_receiver.initalize_receiver(9999)

    while True:
        frame = sock_receiver.receive_array()

        if time() - start >= 1:
            fps = count
            count = 0
            start = time()

        count += 1

        boxes, faces = fast_mtcnn([frame])

        if len(faces):
            extracted_faces = []
            
            for face in faces:
                arr = asarray(face)
                if arr.shape[0] > 0 and arr.shape[1] > 0:
                    img = Image.fromarray(face)
                    img_resize = img.resize((224, 224))
                    face_arr = asarray(img_resize)

                    extracted_faces.append(face_arr)

            if len(extracted_faces) == 0:
                continue
            
            embedding_X_train = []

            for face in extracted_faces:
                embedding = get_embedding(embedding_model, face)
                embedding_X_train.append(embedding)

            embedding_X_train = asarray(embedding_X_train)

            norm = Normalizer(norm='l2')
            trainX = norm.transform(embedding_X_train)

            preds = svc_model.predict(trainX)
            pred_probs = svc_model.predict_proba(trainX)
            pred_names = encoder.inverse_transform(preds)

            res = []

            for pred_prob, pred_name, box in zip(pred_probs, pred_names, boxes[0]):
                accuracy = pred_prob[np.argmax(pred_prob)]*100

                if accuracy > 75:
                    text = '{} {:.2f}%'.format(pred_name, accuracy)
                else:
                    text = 'Unknown'

                res.append([text, box])

            sock_receiver.response_to_sender([res[0][0]])

                # print(text)

        print('FPS: {}'.format(fps))
        print('-'*50)

if __name__ == "__main__":
    main()