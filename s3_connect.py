# import boto3

# AWS_ACCESS_KEY = "AKIAU6GDZ45TZ3RJRWXO"
# AWS_SECRET_KEY = "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"
# AWS_S3_BUCKET_NAME = "fptestbuckett"
# AWS_REGION = "ap-southeast-2"
# LOCAL_FILE = 'test.txt'
# NAME_FOR_S3 = 'test.txt'

# def main():
#     print('in main method')

#     session = boto3.Session(
#         aws_access_key_id=AWS_ACCESS_KEY,
#         aws_secret_access_key=AWS_SECRET_KEY,
#         region_name=AWS_REGION
#     )

#     s3_client = session.client('s3')

#     response = s3_client.upload_file(LOCAL_FILE, AWS_S3_BUCKET_NAME, NAME_FOR_S3)

#     print(f'upload file response: {response}')

# if __name__ == '__main__':
#     main()


import boto3
import os

AWS_ACCESS_KEY = "AKIAU6GDZ45TZ3RJRWXO"
AWS_SECRET_KEY = "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"
AWS_S3_BUCKET_NAME = "fptestbuckett"
AWS_REGION = "ap-southeast-2"
LOCAL_DIRECTORY = 'dataset_split'  # Thay bằng đường dẫn đến thư mục của bạn

def upload_directory_to_s3(directory, bucket_name, s3_client):
    for root, dirs, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory)
            s3_path = relative_path.replace("\\", "/")  # S3 sử dụng dấu gạch chéo (/)
            try:
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f'Successfully uploaded {s3_path} to {bucket_name}')
            except Exception as e:
                print(f'Failed to upload {s3_path}: {e}')

def main():
    print('in main method')

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    s3_client = session.client('s3')

    upload_directory_to_s3(LOCAL_DIRECTORY, AWS_S3_BUCKET_NAME, s3_client)

if __name__ == '__main__':
    main()
