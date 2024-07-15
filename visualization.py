import pandas as pd
import math
import cv2
import numpy as np

# 视频文件路径
video_path = 'path_to_your_video.mp4'
# 假设CSV文件的路径
key_csv_path = 'testnew.csv'
csv_path = 'testnew.csv'
# 读取CSV文件
key_df = pd.read_csv(key_csv_path)
df = pd.read_csv(csv_path)


frame_count = 0  # 当前帧计数器


key_x, key_y = key_df.iloc[0:, 2], key_df.iloc[0:, 3]
x, y = df.iloc[0:, 2], df.iloc[0:, 3]

# 打开视频文件
cap = cv2.VideoCapture(video_path)
# 检查是否成功打开视频文件
if not cap.isOpened():
    print("Error opening video file")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # 在小球位置画一个红色的点
    cv2.circle(frame, (key_x[frame_count], key_y[frame_count]), 10, (0, 0, 255), -1)
    cv2.circle(frame, (x[frame_count], y[frame_count]), 10, (0, 255, 255), -1)
    # 显示帧
    # cv2.imshow('Video with Ball Marker', frame)

    # 等待按键，如果按下 'q' 则退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# 清理
cap.release()
cv2.destroyAllWindows()