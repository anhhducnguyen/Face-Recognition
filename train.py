from os import listdir
from os.path import isdir, exists
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Function to detect new directories
def detect_new_dirs(dataset_dir, pickle_file):
    def load_existing_dirs(pickle_file):
        if exists(pickle_file):
            with open(pickle_file, 'rb') as file:
                return pickle.load(file)
        else:
            return set()

    def save_existing_dirs(pickle_file, directories):
        with open(pickle_file, 'wb') as file:
            pickle.dump(directories, file)

    # Load existing directories
    existing_dirs = load_existing_dirs(pickle_file)

    # Load dataset and detect new directories
    new_dirs = set(listdir(dataset_dir)) - existing_dirs
    if new_dirs:
        print(f"New directories found in {dataset_dir}: {new_dirs}")
        # Update the existing directories with the new ones
        updated_dirs = existing_dirs.union(new_dirs)
        save_existing_dirs(pickle_file, updated_dirs)
        return new_dirs
    else:
        print(f"No new data found in {dataset_dir}.")
        return None

# Extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# Load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        face = extract_face(path)
        if face is None:
            continue
        faces.append(face)
    return faces

# Load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, new_dirs=None):
    X, y = list(), list()
    subdirs = new_dirs if new_dirs else listdir(directory)
    for subdir in subdirs:
        path = directory + '/' + subdir + '/'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return X, y

# Get embedding for a face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Main function to load data, process it, and train the model
def main():
    dataset_train_dir = 'dataset_split/train/'
    dataset_val_dir = 'dataset_split/val/'
    pickle_train_file = 'check_data/train_existing_dirs.pkl'
    pickle_val_file = 'check_data/val_existing_dirs.pkl'

    # Detect new directories
    new_train_dirs = detect_new_dirs(dataset_train_dir, pickle_train_file)
    new_val_dirs = detect_new_dirs(dataset_val_dir, pickle_val_file)
    
    # Load new data
    new_trainX, new_trainy = load_dataset(dataset_train_dir, new_train_dirs) if new_train_dirs else ([], [])
    new_valX, new_valy = load_dataset(dataset_val_dir, new_val_dirs) if new_val_dirs else ([], [])

    # Create FaceNet embedder
    embedder = FaceNet()
    model = embedder.model

    # Convert faces in train and test set to embeddings
    new_embeddings_trainX = [get_embedding(model, face) for face in new_trainX]
    new_embeddings_valX = [get_embedding(model, face) for face in new_valX]

    # Load previous data if available
    old_embeddings_trainX, old_trainy = [], []
    old_embeddings_valX, old_valy = [], []

    if os.path.exists('train_emb_lab/train_embeddings.pkl') and os.path.exists('train_emb_lab/train_labels.pkl'):
        with open('train_emb_lab/train_embeddings.pkl', 'rb') as file:
            old_embeddings_trainX = pickle.load(file)
        with open('train_emb_lab/train_labels.pkl', 'rb') as file:
            old_trainy = pickle.load(file)
        
        # Combine old and new embeddings
        combined_embeddings_trainX_list = old_embeddings_trainX.tolist()
        combined_embeddings_trainX_list.extend(new_embeddings_trainX)
        combined_embeddings_trainX = np.asarray(combined_embeddings_trainX_list)

        combined_trainy = old_trainy + new_trainy

        # Save combined data
        with open('train_emb_lab/train_embeddings.pkl', 'wb') as file:
            pickle.dump(combined_embeddings_trainX, file)
        with open('train_emb_lab/train_labels.pkl', 'wb') as file:
            pickle.dump(combined_trainy, file)
    else:
        combined_embeddings_trainX = np.asarray(new_embeddings_trainX)
        combined_trainy = new_trainy
        with open('train_emb_lab/train_embeddings.pkl', 'wb') as file:
            pickle.dump(combined_embeddings_trainX, file)
        with open('train_emb_lab/train_labels.pkl', 'wb') as file:
            pickle.dump(combined_trainy, file)

    if os.path.exists('val_emb_lab/val_embeddings.pkl') and os.path.exists('val_emb_lab/val_labels.pkl'):
        with open('val_emb_lab/val_embeddings.pkl', 'rb') as file:
            old_embeddings_valX = pickle.load(file)
        with open('val_emb_lab/val_labels.pkl', 'rb') as file:
            old_valy = pickle.load(file)
        
        # Combine old and new embeddings
        combined_embeddings_valX_list = old_embeddings_valX.tolist()
        combined_embeddings_valX_list.extend(new_embeddings_valX)
        combined_embeddings_valX = np.asarray(combined_embeddings_valX_list)

        combined_valy = old_valy + new_valy

        # Save combined data
        with open('val_emb_lab/val_embeddings.pkl', 'wb') as file:
            pickle.dump(combined_embeddings_valX, file)
        with open('val_emb_lab/val_labels.pkl', 'wb') as file:
            pickle.dump(combined_valy, file)
    else:
        combined_embeddings_valX = np.asarray(new_embeddings_valX)
        combined_valy = new_valy
        with open('val_emb_lab/val_embeddings.pkl', 'wb') as file:
            pickle.dump(combined_embeddings_valX, file)
        with open('val_emb_lab/val_labels.pkl', 'wb') as file:
            pickle.dump(combined_valy, file)
    
    # Normalize input vectors
    in_encoder = Normalizer(norm='l2')
    combined_embeddings_trainX = in_encoder.transform(np.asarray(combined_embeddings_trainX))
    combined_embeddings_valX = in_encoder.transform(np.asarray(combined_embeddings_valX))

    # Label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(combined_trainy)
    combined_trainy = out_encoder.transform(combined_trainy)
    combined_valy = out_encoder.transform(combined_valy)

    # Train SVM model
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(combined_embeddings_trainX, combined_trainy)

    # Save the trained model and out_encoder for use in realtime_face_recognition
    with open('svm_model/svm_model.pkl', 'wb') as model_file:
        pickle.dump(svm_model, model_file)
    with open('svm_model/out_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(out_encoder, encoder_file)

    # Evaluate model on test set
    y_pred = svm_model.predict(combined_embeddings_valX)
    print("\nConfusion Matrix:")
    print(confusion_matrix(combined_valy, y_pred))
    print("\nClassification Report:")
    print(classification_report(combined_valy, y_pred))

if __name__ == "__main__":
    main()



# from sklearn.externals import joblib
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# def train():
#     # Load dataset
#     iris = datasets.load_iris()
#     X, y = iris.data, iris.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     # Train model
#     clf = RandomForestClassifier()
#     clf.fit(X_train, y_train)

#     # Save the model
#     joblib.dump(clf, '/opt/ml/model/model.joblib')

# if __name__ == '__main__':
#     train()


# import boto3
# from os import listdir, makedirs
# from os.path import isdir, exists, join
# from PIL import Image
# import numpy as np
# from mtcnn.mtcnn import MTCNN
# from keras_facenet import FaceNet
# from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import pickle
# import os

# # AWS S3 Configuration
# AWS_ACCESS_KEY = "AKIAU6GDZ45TZ3RJRWXO"
# AWS_SECRET_KEY = "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"
# AWS_S3_BUCKET_NAME = "fptestbuckett"
# AWS_REGION = "ap-southeast-2"
# LOCAL_DATASET_DIR = 'dataset_split'

# # Function to download directory from S3
# def download_directory_from_s3(bucket_name, remote_directory, local_directory, s3_client):
#     paginator = s3_client.get_paginator('list_objects_v2')
#     for result in paginator.paginate(Bucket=bucket_name, Prefix=remote_directory):
#         for content in result.get('Contents', []):
#             key = content['Key']
#             if key.endswith('/'):
#                 continue
#             local_path = join(local_directory, key[len(remote_directory):])
#             local_dir = os.path.dirname(local_path)
#             if not exists(local_dir):
#                 makedirs(local_dir)
#             s3_client.download_file(bucket_name, key, local_path)
#             print(f'Successfully downloaded {key} to {local_path}')

# # Function to detect new directories
# def detect_new_dirs(dataset_dir, pickle_file):
#     def load_existing_dirs(pickle_file):
#         if exists(pickle_file):
#             with open(pickle_file, 'rb') as file:
#                 return pickle.load(file)
#         else:
#             return set()

#     def save_existing_dirs(pickle_file, directories):
#         with open(pickle_file, 'wb') as file:
#             pickle.dump(directories, file)

#     # Load existing directories
#     existing_dirs = load_existing_dirs(pickle_file)

#     # Load dataset and detect new directories
#     new_dirs = set(listdir(dataset_dir)) - existing_dirs
#     if new_dirs:
#         print(f"New directories found in {dataset_dir}: {new_dirs}")
#         # Update the existing directories with the new ones
#         updated_dirs = existing_dirs.union(new_dirs)
#         save_existing_dirs(pickle_file, updated_dirs)
#         return new_dirs
#     else:
#         print(f"No new data found in {dataset_dir}.")
#         return None

# # Extract a single face from a given photograph
# def extract_face(filename, required_size=(160, 160)):
#     image = Image.open(filename)
#     image = image.convert('RGB')
#     pixels = np.asarray(image)
#     detector = MTCNN()
#     results = detector.detect_faces(pixels)
#     if len(results) == 0:
#         return None
#     x1, y1, width, height = results[0]['box']
#     x1, y1 = abs(x1), abs(y1)
#     x2, y2 = x1 + width, y1 + height
#     face = pixels[y1:y2, x1:x2]
#     image = Image.fromarray(face)
#     image = image.resize(required_size)
#     face_array = np.asarray(image)
#     return face_array

# # Load images and extract faces for all images in a directory
# def load_faces(directory):
#     faces = list()
#     for filename in listdir(directory):
#         path = directory + '/' + filename
#         face = extract_face(path)
#         if face is None:
#             continue
#         faces.append(face)
#     return faces

# # Load a dataset that contains one subdir for each class that in turn contains images
# def load_dataset(directory, new_dirs=None):
#     X, y = list(), list()
#     subdirs = new_dirs if new_dirs else listdir(directory)
#     for subdir in subdirs:
#         path = directory + '/' + subdir + '/'
#         if not isdir(path):
#             continue
#         faces = load_faces(path)
#         labels = [subdir for _ in range(len(faces))]
#         print('>loaded %d examples for class: %s' % (len(faces), subdir))
#         X.extend(faces)
#         y.extend(labels)
#     return X, y

# # Get embedding for a face
# def get_embedding(model, face_pixels):
#     face_pixels = face_pixels.astype('float32')
#     mean, std = face_pixels.mean(), face_pixels.std()
#     face_pixels = (face_pixels - mean) / std
#     samples = np.expand_dims(face_pixels, axis=0)
#     yhat = model.predict(samples)
#     return yhat[0]

# # Main function to load data, process it, and train the model
# def main():
#     # Initialize AWS S3 client
#     session = boto3.Session(
#         aws_access_key_id=AWS_ACCESS_KEY,
#         aws_secret_access_key=AWS_SECRET_KEY,
#         region_name=AWS_REGION
#     )
#     s3_client = session.client('s3')

#     # Download datasets from S3
#     download_directory_from_s3(AWS_S3_BUCKET_NAME, 'train/', join(LOCAL_DATASET_DIR, 'train'), s3_client)
#     download_directory_from_s3(AWS_S3_BUCKET_NAME, 'val/', join(LOCAL_DATASET_DIR, 'val'), s3_client)

#     dataset_train_dir = join(LOCAL_DATASET_DIR, 'train')
#     dataset_val_dir = join(LOCAL_DATASET_DIR, 'val')
#     pickle_train_file = 'check_data/train_existing_dirs.pkl'
#     pickle_val_file = 'check_data/val_existing_dirs.pkl'

#     # Detect new directories
#     new_train_dirs = detect_new_dirs(dataset_train_dir, pickle_train_file)
#     new_val_dirs = detect_new_dirs(dataset_val_dir, pickle_val_file)
    
#     # Load new data
#     new_trainX, new_trainy = load_dataset(dataset_train_dir, new_train_dirs) if new_train_dirs else ([], [])
#     new_valX, new_valy = load_dataset(dataset_val_dir, new_val_dirs) if new_val_dirs else ([], [])

#     # Create FaceNet embedder
#     embedder = FaceNet()
#     model = embedder.model

#     # Convert faces in train and test set to embeddings
#     new_embeddings_trainX = [get_embedding(model, face) for face in new_trainX]
#     new_embeddings_valX = [get_embedding(model, face) for face in new_valX]

#     # Load previous data if available
#     old_embeddings_trainX, old_trainy = [], []
#     old_embeddings_valX, old_valy = [], []

#     if os.path.exists('train_emb_lab/train_embeddings.pkl') and os.path.exists('train_emb_lab/train_labels.pkl'):
#         with open('train_emb_lab/train_embeddings.pkl', 'rb') as file:
#             old_embeddings_trainX = pickle.load(file)
#         with open('train_emb_lab/train_labels.pkl', 'rb') as file:
#             old_trainy = pickle.load(file)
        
#         # Combine old and new embeddings
#         combined_embeddings_trainX_list = old_embeddings_trainX.tolist()
#         combined_embeddings_trainX_list.extend(new_embeddings_trainX)
#         combined_embeddings_trainX = np.asarray(combined_embeddings_trainX_list)

#         combined_trainy = old_trainy + new_trainy

#         # Save combined data
#         with open('train_emb_lab/train_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_trainX, file)
#         with open('train_emb_lab/train_labels.pkl', 'wb') as file:
#             pickle.dump(combined_trainy, file)
#     else:
#         combined_embeddings_trainX = np.asarray(new_embeddings_trainX)
#         combined_trainy = new_trainy
#         with open('train_emb_lab/train_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_trainX, file)
#         with open('train_emb_lab/train_labels.pkl', 'wb') as file:
#             pickle.dump(combined_trainy, file)

#     if os.path.exists('val_emb_lab/val_embeddings.pkl') and os.path.exists('val_emb_lab/val_labels.pkl'):
#         with open('val_emb_lab/val_embeddings.pkl', 'rb') as file:
#             old_embeddings_valX = pickle.load(file)
#         with open('val_emb_lab/val_labels.pkl', 'rb') as file:
#             old_valy = pickle.load(file)
        
#         # Combine old and new embeddings
#         combined_embeddings_valX_list = old_embeddings_valX.tolist()
#         combined_embeddings_valX_list.extend(new_embeddings_valX)
#         combined_embeddings_valX = np.asarray(combined_embeddings_valX_list)

#         combined_valy = old_valy + new_valy

#         # Save combined data
#         with open('val_emb_lab/val_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_valX, file)
#         with open('val_emb_lab/val_labels.pkl', 'wb') as file:
#             pickle.dump(combined_valy, file)
#     else:
#         combined_embeddings_valX = np.asarray(new_embeddings_valX)
#         combined_valy = new_valy
#         with open('val_emb_lab/val_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_valX, file)
#         with open('val_emb_lab/val_labels.pkl', 'wb') as file:
#             pickle.dump(combined_valy, file)
    
#     # Normalize input vectors
#     in_encoder = Normalizer(norm='l2')
#     combined_embeddings_trainX = in_encoder.transform(np.asarray(combined_embeddings_trainX))
#     combined_embeddings_valX = in_encoder.transform(np.asarray(combined_embeddings_valX))

#     # Label encode targets
#     out_encoder = LabelEncoder()
#     out_encoder.fit(combined_trainy)
#     combined_trainy = out_encoder.transform(combined_trainy)
#     combined_valy = out_encoder.transform(combined_valy)

#     # Train SVM model
#     svm_model = SVC(kernel='linear', probability=True)
#     svm_model.fit(combined_embeddings_trainX, combined_trainy)

#     # Save the trained model and out_encoder for use in realtime_face_recognition
#     with open('svm_model/svm_model.pkl', 'wb') as model_file:
#         pickle.dump(svm_model, model_file)
#     with open('svm_model/out_encoder.pkl', 'wb') as encoder_file:
#         pickle.dump(out_encoder, encoder_file)

#     # Evaluate model on test set
#     y_pred = svm_model.predict(combined_embeddings_valX)
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(combined_valy, y_pred))
#     print("\nClassification Report:")
#     print(classification_report(combined_valy, y_pred))

# if __name__ == "__main__":
#     main()



# import os
# from os import listdir
# from os.path import isdir, exists
# from PIL import Image
# import numpy as np
# from mtcnn.mtcnn import MTCNN
# from keras_facenet import FaceNet
# from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import pickle
# import boto3
# import io

# # AWS S3 Configuration
# AWS_ACCESS_KEY = "AKIAU6GDZ45TZ3RJRWXO"
# AWS_SECRET_KEY = "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"
# AWS_S3_BUCKET_NAME = "fptestbuckett"
# AWS_REGION = "ap-southeast-2"

# # Function to detect new directories in S3
# def detect_new_dirs(s3_client, bucket_name, prefix, pickle_file):
#     def load_existing_dirs(pickle_file):
#         if exists(pickle_file):
#             with open(pickle_file, 'rb') as file:
#                 return pickle.load(file)
#         else:
#             return set()

#     def save_existing_dirs(pickle_file, directories):
#         with open(pickle_file, 'wb') as file:
#             pickle.dump(directories, file)

#     # Load existing directories
#     existing_dirs = load_existing_dirs(pickle_file)

#     # List directories in S3
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
#     new_dirs = set()
#     for content in response.get('CommonPrefixes', []):
#         dir_name = content.get('Prefix').split('/')[-2]
#         if dir_name:
#             new_dirs.add(dir_name)

#     if new_dirs:
#         new_dirs -= existing_dirs
#         if new_dirs:
#             print(f"New directories found in {prefix}: {new_dirs}")
#             # Update the existing directories with the new ones
#             updated_dirs = existing_dirs.union(new_dirs)
#             save_existing_dirs(pickle_file, updated_dirs)
#             return new_dirs
#     else:
#         print(f"No new data found in {prefix}.")
#         return None

# # Extract a single face from a given photograph
# def extract_face(image_bytes, required_size=(160, 160)):
#     image = Image.open(io.BytesIO(image_bytes))
#     image = image.convert('RGB')
#     pixels = np.asarray(image)
#     detector = MTCNN()
#     results = detector.detect_faces(pixels)
#     if len(results) == 0:
#         return None
#     x1, y1, width, height = results[0]['box']
#     x1, y1 = abs(x1), abs(y1)
#     x2, y2 = x1 + width, y1 + height
#     face = pixels[y1:y2, x1:x2]
#     image = Image.fromarray(face)
#     image = image.resize(required_size)
#     face_array = np.asarray(image)
#     return face_array

# # Load images and extract faces for all images in a directory in S3
# def load_faces_from_s3(s3_client, bucket_name, prefix):
#     faces = list()
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
#     for obj in response.get('Contents', []):
#         file_key = obj['Key']
#         if file_key.endswith('.jpg') or file_key.endswith('.png'):
#             file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
#             file_content = file_obj['Body'].read()
#             face = extract_face(file_content)
#             if face is None:
#                 continue
#             faces.append(face)
#     return faces

# # Load a dataset that contains one subdir for each class that in turn contains images
# def load_dataset_from_s3(s3_client, bucket_name, prefix, new_dirs=None):
#     X, y = list(), list()
#     subdirs = new_dirs if new_dirs else []
#     for subdir in subdirs:
#         path = f'{prefix}/{subdir}/'
#         faces = load_faces_from_s3(s3_client, bucket_name, path)
#         labels = [subdir for _ in range(len(faces))]
#         print('>loaded %d examples for class: %s' % (len(faces), subdir))
#         X.extend(faces)
#         y.extend(labels)
#     return X, y

# # Get embedding for a face
# def get_embedding(model, face_pixels):
#     face_pixels = face_pixels.astype('float32')
#     mean, std = face_pixels.mean(), face_pixels.std()
#     face_pixels = (face_pixels - mean) / std
#     samples = np.expand_dims(face_pixels, axis=0)
#     yhat = model.predict(samples)
#     return yhat[0]

# # Main function to load data, process it, and train the model
# def main():
#     s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
#     dataset_train_prefix = 'train'
#     dataset_val_prefix = 'val'
#     pickle_train_file = 'check_data/train_existing_dirs.pkl'
#     pickle_val_file = 'check_data/val_existing_dirs.pkl'

#     # Detect new directories
#     new_train_dirs = detect_new_dirs(s3_client, AWS_S3_BUCKET_NAME, dataset_train_prefix, pickle_train_file)
#     new_val_dirs = detect_new_dirs(s3_client, AWS_S3_BUCKET_NAME, dataset_val_prefix, pickle_val_file)
    
#     # Load new data
#     new_trainX, new_trainy = load_dataset_from_s3(s3_client, AWS_S3_BUCKET_NAME, dataset_train_prefix, new_train_dirs) if new_train_dirs else ([], [])
#     new_valX, new_valy = load_dataset_from_s3(s3_client, AWS_S3_BUCKET_NAME, dataset_val_prefix, new_val_dirs) if new_val_dirs else ([], [])

#     # Create FaceNet embedder
#     embedder = FaceNet()
#     model = embedder.model

#     # Convert faces in train and test set to embeddings
#     new_embeddings_trainX = [get_embedding(model, face) for face in new_trainX]
#     new_embeddings_valX = [get_embedding(model, face) for face in new_valX]

#     # Load previous data if available
#     old_embeddings_trainX, old_trainy = [], []
#     old_embeddings_valX, old_valy = [], []

#     if os.path.exists('train_emb_lab/train_embeddings.pkl') and os.path.exists('train_emb_lab/train_labels.pkl'):
#         with open('train_emb_lab/train_embeddings.pkl', 'rb') as file:
#             old_embeddings_trainX = pickle.load(file)
#         with open('train_emb_lab/train_labels.pkl', 'rb') as file:
#             old_trainy = pickle.load(file)
        
#         # Combine old and new embeddings
#         combined_embeddings_trainX_list = old_embeddings_trainX.tolist()
#         combined_embeddings_trainX_list.extend(new_embeddings_trainX)
#         combined_embeddings_trainX = np.asarray(combined_embeddings_trainX_list)

#         combined_trainy = old_trainy + new_trainy

#         # Save combined data
#         with open('train_emb_lab/train_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_trainX, file)
#         with open('train_emb_lab/train_labels.pkl', 'wb') as file:
#             pickle.dump(combined_trainy, file)
#     else:
#         combined_embeddings_trainX = np.asarray(new_embeddings_trainX)
#         combined_trainy = new_trainy
#         with open('train_emb_lab/train_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_trainX, file)
#         with open('train_emb_lab/train_labels.pkl', 'wb') as file:
#             pickle.dump(combined_trainy, file)

#     if os.path.exists('val_emb_lab/val_embeddings.pkl') and os.path.exists('val_emb_lab/val_labels.pkl'):
#         with open('val_emb_lab/val_embeddings.pkl', 'rb') as file:
#             old_embeddings_valX = pickle.load(file)
#         with open('val_emb_lab/val_labels.pkl', 'rb') as file:
#             old_valy = pickle.load(file)
        
#         # Combine old and new embeddings
#         combined_embeddings_valX_list = old_embeddings_valX.tolist()
#         combined_embeddings_valX_list.extend(new_embeddings_valX)
#         combined_embeddings_valX = np.asarray(combined_embeddings_valX_list)

#         combined_valy = old_valy + new_valy

#         # Save combined data
#         with open('val_emb_lab/val_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_valX, file)
#         with open('val_emb_lab/val_labels.pkl', 'wb') as file:
#             pickle.dump(combined_valy, file)
#     else:
#         combined_embeddings_valX = np.asarray(new_embeddings_valX)
#         combined_valy = new_valy
#         with open('val_emb_lab/val_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_valX, file)
#         with open('val_emb_lab/val_labels.pkl', 'wb') as file:
#             pickle.dump(combined_valy, file)
    
#     # Normalize input vectors
#     in_encoder = Normalizer(norm='l2')
#     combined_embeddings_trainX = in_encoder.transform(np.asarray(combined_embeddings_trainX))
#     combined_embeddings_valX = in_encoder.transform(np.asarray(combined_embeddings_valX))

#     # Label encode targets
#     out_encoder = LabelEncoder()
#     out_encoder.fit(combined_trainy)
#     combined_trainy = out_encoder.transform(combined_trainy)
#     combined_valy = out_encoder.transform(combined_valy)

#     # Train SVM model
#     svm_model = SVC(kernel='linear', probability=True)
#     svm_model.fit(combined_embeddings_trainX, combined_trainy)

#     # Save the trained model and out_encoder for use in realtime_face_recognition
#     with open('svm_model/svm_model.pkl', 'wb') as model_file:
#         pickle.dump(svm_model, model_file)
#     with open('svm_model/out_encoder.pkl', 'wb') as encoder_file:
#         pickle.dump(out_encoder, encoder_file)

#     # Evaluate model on test set
#     y_pred = svm_model.predict(combined_embeddings_valX)
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(combined_valy, y_pred))
#     print("\nClassification Report:")
#     print(classification_report(combined_valy, y_pred))

# if __name__ == "__main__":
#     main()


# import os
# from os.path import exists
# from PIL import Image
# import numpy as np
# from mtcnn.mtcnn import MTCNN
# from keras_facenet import FaceNet
# from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import pickle
# import boto3
# import io

# # AWS S3 Configuration
# AWS_ACCESS_KEY = "AKIAU6GDZ45TZ3RJRWXO"
# AWS_SECRET_KEY = "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"
# AWS_S3_BUCKET_NAME = "fptestbuckett"
# AWS_REGION = "ap-southeast-2"

# # Function to detect new directories in S3
# def detect_new_dirs(s3_client, bucket_name, prefix, pickle_file):
#     def load_existing_dirs(pickle_file):
#         if exists(pickle_file):
#             with open(pickle_file, 'rb') as file:
#                 return pickle.load(file)
#         else:
#             return set()

#     def save_existing_dirs(pickle_file, directories):
#         with open(pickle_file, 'wb') as file:
#             pickle.dump(directories, file)

#     # Load existing directories
#     existing_dirs = load_existing_dirs(pickle_file)

#     # List objects in S3
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
#     new_dirs = set()
#     for content in response.get('CommonPrefixes', []):
#         dir_name = content.get('Prefix').split('/')[-2]
#         if dir_name:
#             new_dirs.add(dir_name)

#     if new_dirs:
#         new_dirs -= existing_dirs
#         if new_dirs:
#             print(f"New directories found in {prefix}: {new_dirs}")
#             # Update existing directories with the new ones
#             updated_dirs = existing_dirs.union(new_dirs)
#             save_existing_dirs(pickle_file, updated_dirs)
#             return new_dirs
#     else:
#         print(f"No new data found in {prefix}.")
#         return None

# # Extract a single face from a given photograph
# def extract_face(image_bytes, required_size=(160, 160)):
#     image = Image.open(io.BytesIO(image_bytes))
#     image = image.convert('RGB')
#     pixels = np.asarray(image)
#     detector = MTCNN()
#     results = detector.detect_faces(pixels)
#     if len(results) == 0:
#         return None
#     x1, y1, width, height = results[0]['box']
#     x1, y1 = abs(x1), abs(y1)
#     x2, y2 = x1 + width, y1 + height
#     face = pixels[y1:y2, x1:x2]
#     image = Image.fromarray(face)
#     image = image.resize(required_size)
#     face_array = np.asarray(image)
#     return face_array

# # Load images and extract faces for all images in a directory in S3
# def load_faces_from_s3(s3_client, bucket_name, prefix):
#     faces = list()
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
#     for obj in response.get('Contents', []):
#         file_key = obj['Key']
#         if file_key.endswith('.jpg') or file_key.endswith('.png'):
#             file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
#             file_content = file_obj['Body'].read()
#             face = extract_face(file_content)
#             if face is None:
#                 continue
#             faces.append(face)
#     return faces

# # Load a dataset that contains one subdir for each class that in turn contains images
# def load_dataset_from_s3(s3_client, bucket_name, prefix, new_dirs=None):
#     X, y = list(), list()
#     subdirs = new_dirs if new_dirs else []
#     for subdir in subdirs:
#         path = f'{prefix}/{subdir}/'
#         faces = load_faces_from_s3(s3_client, bucket_name, path)
#         labels = [subdir for _ in range(len(faces))]
#         print(f'>loaded {len(faces)} examples for class: {subdir}')
#         X.extend(faces)
#         y.extend(labels)
#     return X, y

# # Get embedding for a face
# def get_embedding(model, face_pixels):
#     face_pixels = face_pixels.astype('float32')
#     mean, std = face_pixels.mean(), face_pixels.std()
#     face_pixels = (face_pixels - mean) / std
#     samples = np.expand_dims(face_pixels, axis=0)
#     yhat = model.predict(samples)
#     return yhat[0]

# # Main function to load data, process it, and train the model
# def main():
#     s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
#     dataset_train_prefix = 'train'
#     dataset_val_prefix = 'val'
#     pickle_train_file = 'train_existing_dirs.pkl'
#     pickle_val_file = 'val_existing_dirs.pkl'

#     # Detect new directories
#     new_train_dirs = detect_new_dirs(s3_client, AWS_S3_BUCKET_NAME, dataset_train_prefix, pickle_train_file)
#     new_val_dirs = detect_new_dirs(s3_client, AWS_S3_BUCKET_NAME, dataset_val_prefix, pickle_val_file)
    
#     # Load new data
#     new_trainX, new_trainy = load_dataset_from_s3(s3_client, AWS_S3_BUCKET_NAME, dataset_train_prefix, new_train_dirs) if new_train_dirs else ([], [])
#     new_valX, new_valy = load_dataset_from_s3(s3_client, AWS_S3_BUCKET_NAME, dataset_val_prefix, new_val_dirs) if new_val_dirs else ([], [])

#     # Create FaceNet embedder
#     embedder = FaceNet()
#     model = embedder.model

#     # Convert faces in train and test set to embeddings
#     new_embeddings_trainX = [get_embedding(model, face) for face in new_trainX]
#     new_embeddings_valX = [get_embedding(model, face) for face in new_valX]

#     # Load previous data if available
#     old_embeddings_trainX, old_trainy = [], []
#     old_embeddings_valX, old_valy = [], []

#     if exists('train_embeddings.pkl') and exists('train_labels.pkl'):
#         with open('train_embeddings.pkl', 'rb') as file:
#             old_embeddings_trainX = pickle.load(file)
#         with open('train_labels.pkl', 'rb') as file:
#             old_trainy = pickle.load(file)
        
#         # Combine old and new embeddings
#         combined_embeddings_trainX_list = old_embeddings_trainX.tolist()
#         combined_embeddings_trainX_list.extend(new_embeddings_trainX)
#         combined_embeddings_trainX = np.asarray(combined_embeddings_trainX_list)

#         combined_trainy = old_trainy + new_trainy

#         # Save combined data
#         with open('train_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_trainX, file)
#         with open('train_labels.pkl', 'wb') as file:
#             pickle.dump(combined_trainy, file)
#     else:
#         combined_embeddings_trainX = np.asarray(new_embeddings_trainX)
#         combined_trainy = new_trainy
#         with open('train_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_trainX, file)
#         with open('train_labels.pkl', 'wb') as file:
#             pickle.dump(combined_trainy, file)

#     if exists('val_embeddings.pkl') and exists('val_labels.pkl'):
#         with open('val_embeddings.pkl', 'rb') as file:
#             old_embeddings_valX = pickle.load(file)
#         with open('val_labels.pkl', 'rb') as file:
#             old_valy = pickle.load(file)
        
#         # Combine old and new embeddings
#         combined_embeddings_valX_list = old_embeddings_valX.tolist()
#         combined_embeddings_valX_list.extend(new_embeddings_valX)
#         combined_embeddings_valX = np.asarray(combined_embeddings_valX_list)

#         combined_valy = old_valy + new_valy

#         # Save combined data
#         with open('val_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_valX, file)
#         with open('val_labels.pkl', 'wb') as file:
#             pickle.dump(combined_valy, file)
#     else:
#         combined_embeddings_valX = np.asarray(new_embeddings_valX)
#         combined_valy = new_valy
#         with open('val_embeddings.pkl', 'wb') as file:
#             pickle.dump(combined_embeddings_valX, file)
#         with open('val_labels.pkl', 'wb') as file:
#             pickle.dump(combined_valy, file)
    
#     # Normalize input vectors
#     in_encoder = Normalizer(norm='l2')
#     combined_embeddings_trainX = in_encoder.transform(np.asarray(combined_embeddings_trainX))

#     combined_embeddings_valX = in_encoder.transform(np.asarray(combined_embeddings_valX))

#     # Label encode targets
#     out_encoder = LabelEncoder()
#     out_encoder.fit(combined_trainy)
#     trainy = out_encoder.transform(combined_trainy)
#     out_encoder.fit(combined_valy)
#     valy = out_encoder.transform(combined_valy)

#     # Define model
#     model = SVC(kernel='linear', probability=True)

#     # Fit model
#     model.fit(combined_embeddings_trainX, trainy)

#     # Predict
#     yhat_train = model.predict(combined_embeddings_trainX)
#     yhat_val = model.predict(combined_embeddings_valX)

#     # Score
#     score_train = model.score(combined_embeddings_trainX, trainy)
#     score_val = model.score(combined_embeddings_valX, valy)

#     # Classification report and confusion matrix
#     print('Train Accuracy: %.3f' % score_train)
#     print('Validation Accuracy: %.3f' % score_val)
#     print('Train Classification Report:\n', classification_report(trainy, yhat_train))
#     print('Validation Classification Report:\n', classification_report(valy, yhat_val))
#     print('Train Confusion Matrix:\n', confusion_matrix(trainy, yhat_train))
#     print('Validation Confusion Matrix:\n', confusion_matrix(valy, yhat_val))

# if __name__ == "__main__":
#     main()
