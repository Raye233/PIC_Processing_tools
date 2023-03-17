from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import winsound
import argparse
import dlib
import cv2


def wink():
    def eye_aspect_ratio(eye):
        # 计算两组垂直眼界标（x，y）坐标之间的欧几里得距离
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # 计算水平眼界标（x，y）坐标之间的欧几里得距离
        C = dist.euclidean(eye[0], eye[3])

        # 计算眼睛长宽比
        ear = (A + B) / (2.0 * C)

        # 返回眼睛长宽比
        return ear

    # 构造参数解析并解析参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
                    help="path to input video file")
    args = vars(ap.parse_args())

    # 定义两个常数，一个常数表示眼睛的纵横比以指示眨眼，然后定义第二个常数表示眼睛的连续帧数必须低于阈值
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 2

    # 初始化帧计数器和闪烁总数
    COUNTER = 0
    TOTAL = 0

    # 初始化dlib的面部检测器（基于HOG），然后创建面部界标预测器
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # 分别获取左眼和右眼的面部标志的索引
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    #  启动视频流线程
    print("[INFO] starting video stream thread...")
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
    time.sleep(1.0)

    #  循环播放视频流中的帧
    while True:
        # 如果这是文件视频流，那么我们需要检查缓冲区中是否还有剩余的帧要处理
        if fileStream and not vs.more():
            break
        # 从线程视频文件流中抓取帧，调整其大小，然后将其转换为灰度通道）

        frame = vs.read()
        if frame is not None:
            frame = imutils.resize(frame, width=450)  # 调节显示的像素大小 450我的超薄本可以很流畅 实时用dlib识别人脸不卡顿 电脑好的调高画质 效果不错
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            # loop over the face detections 循环人脸检测
            for rect in rects:
                # 确定面部区域的面部界标，然后将面部界标（x，y）坐标转换为NumPy数组
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 提取左眼和右眼坐标，然后使用坐标计算两只眼睛的眼睛纵横比
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # 将两只眼睛的眼睛纵横比平均在一起
                ear = (leftEAR + rightEAR) / 2.0

                # 计算左眼和右眼的凸包，然后可视化每只眼睛
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # 检查眼睛宽高比是否低于眨眼阈值，如果是，则增加眨眼帧计数器
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                # 否则，眼睛纵横比不低于眨眼阈值
                else:
                    # 如果闭上眼睛的次数足够多，则增加眨眼的总数
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1

                    # 重置眼框计数器
                    COUNTER = 0

                # 绘制帧上眨眼的总数以及计算出的帧的眼睛纵横比
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "COUNTER: {}".format(COUNTER), (140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()
