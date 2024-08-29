import cv2
import numpy as np
import os


def resize_and_pad_image(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img_resized = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)

            # 创建1080x1920的黑色背景
            canvas = np.zeros((1080, 1920), dtype=np.uint8)

            # 计算图像居中位置
            top_left_x = (1920 - 800) // 2
            top_left_y = (1080 - 800) // 2

            # 将缩放后的图像粘贴到背景图像上
            canvas[top_left_y:top_left_y + 800, top_left_x:top_left_x + 800] = img_resized

            # 保存处理后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, canvas)


input_folder = "E:/Data/HR_data/HR_gray"
output_folder = "./image"
resize_and_pad_image(input_folder, output_folder)
