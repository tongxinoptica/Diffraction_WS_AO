import time
import os
import simple_pyspin as sp

def check_camera_connection():
    # 尝试打开相机
    try:
        with sp.Camera() as cam:
            cam.init()
            print("Camera initialized successfully.")
            return True
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return False

def capture_images():
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)

    # 打开相机
    with sp.Camera() as cam:
        # 配置相机
        cam.init()
        cam.AcquisitionFrameRateEnable = True
        cam.AcquisitionFrameRate = 1.0  # 设置帧率为1帧每秒

        # 开始采集图像
        cam.start()

        try:
            for i in range(10):  # 设置拍摄次数
                # 获取图像
                image = cam.get_array()

                # 保存图像
                filename = os.path.join(output_dir, f'image_{i:03d}.png')
                sp.save_image(filename, image)

                print(f"Image {i+1} saved as {filename}")

                # 等待1秒
                time.sleep(1)
        except KeyboardInterrupt:
            print("Image capture stopped by user.")
        finally:
            # 停止采集图像
            cam.stop()

if __name__ == "__main__":
    if check_camera_connection():
        print("Camera is connected and ready to use.")
        capture_images()
    else:
        print("Camera is not connected or initialization failed.")
