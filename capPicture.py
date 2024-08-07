# import os
# import cv2
# import numpy as np
# from mtcnn.mtcnn import MTCNN
# from sklearn.model_selection import train_test_split

# # Function to enhance image by applying denoising and smoothing
# def enhance_image(image):
#     # Làm sạch nhiễu bằng Gaussian Blur
#     blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
#     # Tăng cường độ sáng và độ tương phản
#     enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=1.2, beta=10)
    
#     return enhanced_image

# # Function to flip and rotate images to increase dataset size
# def augment_images(images):
#     augmented_images = []
#     for image in images:
        
#         # Flip image vertically
#         flipped_vertical = cv2.flip(image, 1)
#         augmented_images.append(flipped_vertical)
        
#         # Rotate image by 15 degrees clockwise
#         rows, cols, _ = image.shape
#         rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
#         rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
#         augmented_images.append(rotated_image)
        
#         # Rotate image by -15 degrees counterclockwise
#         rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
#         rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
#         augmented_images.append(rotated_image)
#     return augmented_images

# # Function to capture 10 images from the webcam
# def capture_images(name, save_dir='dataset_split'):
#     # Create directories if not exist
#     train_dir = os.path.join(save_dir, 'train', name)
#     val_dir = os.path.join(save_dir, 'val', name)
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)

#     cap = cv2.VideoCapture(0)
#     detector = MTCNN()
#     img_count = 0
#     captured_images = []

#     print("Press 'c' to capture image. Press 'q' to quit.")
#     while img_count < 10:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Display the frame
#         cv2.imshow('Capturing Images', frame)
        
#         # Check for key press
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('c'):
#             results = detector.detect_faces(frame)
#             if results:
#                 x1, y1, width, height = results[0]['box']
#                 x1, y1 = abs(x1), abs(y1)
#                 x2, y2 = x1 + width, y1 + height
#                 face = frame[y1:y2, x1:x2]
#                 face = cv2.resize(face, (160, 160))
#                 # Enhance the captured face image
#                 face = enhance_image(face)
#                 captured_images.append(face)
#                 img_count += 1
#                 print(f"Captured image {img_count}")
#         elif key & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Augment captured images
#     augmented_images = augment_images(captured_images)
    
#     # Split the images into training and validation sets
#     train_images, val_images = train_test_split(augmented_images, test_size=0.33, random_state=42)
    
#     # Save images
#     for i, img in enumerate(train_images):
#         cv2.imwrite(os.path.join(train_dir, f'{name}_{i+1}.jpg'), img)
#     for i, img in enumerate(val_images):
#         cv2.imwrite(os.path.join(val_dir, f'{name}_{i+1}.jpg'), img)

# # Main function
# def main():
#     # Input name
#     name = input("Enter your name: ")
    
#     # Capture images
#     capture_images(name)

#     print("Images captured and saved successfully.")

# if __name__ == "__main__":
#     main()





import boto3
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from sklearn.model_selection import train_test_split

AWS_ACCESS_KEY = "AKIAU6GDZ45TZ3RJRWXO"
AWS_SECRET_KEY = "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"
AWS_S3_BUCKET_NAME = "fptestbuckett"
AWS_REGION = "ap-southeast-2"

# Function to enhance image by applying denoising and smoothing
def enhance_image(image):
    # Làm sạch nhiễu bằng Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Tăng cường độ sáng và độ tương phản
    enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=1.2, beta=10)
    
    return enhanced_image

# Function to flip and rotate images to increase dataset size
def augment_images(images):
    augmented_images = []
    for image in images:
        
        # Flip image vertically
        flipped_vertical = cv2.flip(image, 1)
        augmented_images.append(flipped_vertical)
        
        # Rotate image by 15 degrees clockwise
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        augmented_images.append(rotated_image)
        
        # Rotate image by -15 degrees counterclockwise
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        augmented_images.append(rotated_image)
    return augmented_images

# Function to upload an image to S3
def upload_image_to_s3(image, bucket_name, s3_path, s3_client):
    _, buffer = cv2.imencode('.jpg', image)
    s3_client.put_object(Bucket=bucket_name, Key=s3_path, Body=buffer.tobytes())
    print(f'Successfully uploaded {s3_path} to {bucket_name}')

# Function to capture 10 images from the webcam and upload to S3
def capture_images(name, bucket_name, s3_client):
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    img_count = 0
    captured_images = []

    print("Press 'c' to capture image. Press 'q' to quit.")
    while img_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Capturing Images', frame)
        
        # Check for key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            results = detector.detect_faces(frame)
            if results:
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (160, 160))
                # Enhance the captured face image
                face = enhance_image(face)
                captured_images.append(face)
                img_count += 1
                print(f"Captured image {img_count}")
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Augment captured images
    augmented_images = augment_images(captured_images)
    
    # Split the images into training and validation sets
    train_images, val_images = train_test_split(augmented_images, test_size=0.33, random_state=42)
    
    # Upload images to S3
    for i, img in enumerate(train_images):
        s3_path = f'train/{name}/{name}_{i+1}.jpg'
        upload_image_to_s3(img, bucket_name, s3_path, s3_client)
        
    for i, img in enumerate(val_images):
        s3_path = f'val/{name}/{name}_{i+1}.jpg'
        upload_image_to_s3(img, bucket_name, s3_path, s3_client)

# Main function
def main():
    # Create a session with AWS
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    s3_client = session.client('s3')

    # Input name
    name = input("Enter your name: ")
    
    # Capture images and upload to S3
    capture_images(name, AWS_S3_BUCKET_NAME, s3_client)

    print("Images captured and uploaded successfully.")

if __name__ == "__main__":
    main()
