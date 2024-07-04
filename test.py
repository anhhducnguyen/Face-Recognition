import cv2
import numpy as np

# Đường dẫn tới ảnh của bạn
image_path = 'dataset_split/train/ducanhnguyen/ducanhnguyen_19.jpg'

# Đọc ảnh
image = cv2.imread(image_path)

# Tạo một ảnh nền đen có cùng kích thước với ảnh gốc
black_image = np.zeros_like(image)

# Kích thước và vị trí của bounding box (ví dụ)
start_point = (100, 50)  # Tọa độ góc trên bên trái
end_point = (300, 350)   # Tọa độ góc dưới bên phải

# Màu sắc của bounding box (BGR format)
color = (0, 255, 0)  # Màu xanh lá

# Độ dày của bounding box
thickness = 2

# Vẽ bounding box trên ảnh nền đen
cv2.rectangle(black_image, start_point, end_point, color, thickness)

# Hiển thị ảnh với bounding box
cv2.imshow('Image with Bounding Box', black_image)

# Lưu ảnh với bounding box
output_path = '/mnt/data/image_with_bounding_box.png'
cv2.imwrite(output_path, black_image)

# Đợi nhấn phím bất kỳ để đóng cửa sổ hiển thị
cv2.waitKey(0)
cv2.destroyAllWindows() 
