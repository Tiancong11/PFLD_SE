import cv2
import numpy as np
from models.PFLD_GhostNet import PFLD_GhostNet, PFLD_Ultralight_AuxiliaryNet

# 加载PFLD模型
pfld_model = PFLD_GhostNet()
# auxiliary_net = PFLD_Ultralight_AuxiliaryNet()

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 转换成灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 在人脸上绘制矩形
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 提取眼部区域
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 眼球追踪
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # 在眼部区域绘制矩形
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # 获取眼球中心坐标
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2

            # 在眼球中心绘制一个小圆点
            cv2.circle(frame, (eye_center_x, eye_center_y), 2, (0, 0, 255), 2)

            # 在这里添加眼球追踪算法，例如基于瞳孔检测或深度学习的方法

    # 显示结果
    cv2.imshow('Eye Tracking with PFLD', frame)

    # 按下Esc键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
