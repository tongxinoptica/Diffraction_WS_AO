import cv2
import numpy as np


def feature_matching(image1, image2):
    # SIFT create
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # 使用FLANN匹配特征点
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # 用Lowe's ratio test筛选好的匹配
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 估算变换矩阵
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return matrix
    else:
        return None


def optical_flow_refinement(image1, image2, matrix):
    h, w = image2.shape
    aligned_image = cv2.warpPerspective(image2, matrix, (w, h))
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 使用光流细化对齐
    for y in range(h):
        for x in range(w):
            dx, dy = flow[y, x].astype(np.int32)
            if 0 <= x + dx < w and 0 <= y + dy < h:
                aligned_image[y, x] = aligned_image[y + dy, x + dx]
    return aligned_image

# 加载图像

image_a = cv2.imread('reference_chessboard.jpg', cv2.IMREAD_GRAYSCALE)
image_b = cv2.imread('distorted_chessboard.jpg', cv2.IMREAD_GRAYSCALE)

# 找到变换矩阵
matrix = feature_matching(image_a, image_b)
if matrix is not None:
    # 使用光流细化对齐
    restored_image = optical_flow_refinement(image_a, image_b, matrix)
    # 显示和保存结果
    cv2.imshow('Restored Image', restored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches were found - transformation cannot be done.")
