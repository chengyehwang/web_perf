#!/usr/bin/env python3
import cv2
from tracker import *
import time
import numpy as np
def track(gui=False, **argv):
    frame_start=argv.get('frame_start',0)
    # Create tracker object
    tracker = EuclideanDistTracker()

    cap = cv2.VideoCapture("IMG_3331.mov")
    fps = 10
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)



    # Object detection from Stable camera
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    now_prev = time.time()
    frame_id = 0
    while True:
        frame_id += 1
        ret, frame = cap.read()
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        print(width, height)

        if not ret:
            break

        # calib comp
        roi = frame

        # 1. Object Detection
        #mask = object_detector.apply(roi)
        #_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        can = cv2.Canny(blurred, 50, 200, None, 3)
        
        print(can.shape)
        matrix = np.array(can)
        print(len(matrix))
        value = np.sum(matrix, axis=0)
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w
        value = moving_average(value, 30)
        value = np.clip (value / 3000 * 255, 0, 255)
        print(np.histogram(value))
        print(len(value))

        #can[0,9:] = value
        #can[1,9:] = value
        #can[2,9:] = value

        v_prev = value[0]
        active = 0
        thread = 70
        for x in range(len(value)):
            v = value[x]
            if v_prev > thread and v < thread:
                if x - active > 100:
                    pt1 = (x, 0)
                    pt2 = (x, int(width))
                    cv2.line(roi, pt1, pt2, (0,0,255), 3)
                    active = x
            v_prev = v
        text = str(frame_id)
        cv2.putText(roi, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 1, cv2.LINE_AA)

        _, contours, _ = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if area < 500:
                continue
            if True:
                detections.append([x, y, w, h])

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if gui:
            now = time.time()
            delta = now - now_prev
            now_prev = now
            if delta < 0.1:
                time.sleep(0.1 - delta)

            width = int(width/2)
            height = int(height/2)
            frame = cv2.resize(frame, (width,height))
            gray = cv2.resize(gray, (width, height))
            roi = cv2.resize(roi, (width, height))
            can = cv2.resize(can, (width, height))

            cv2.imshow("Frame", cv2.pyrDown(frame))
            cv2.moveWindow("Frame", 0, 0)
            cv2.imshow("gray", cv2.pyrDown(gray))
            cv2.moveWindow("gray", int(width/2), 0)
            cv2.imshow("roi", cv2.pyrDown(roi))
            cv2.moveWindow("roi", 0, int(height/2)+20)
            cv2.imshow("can", cv2.pyrDown(can))
            cv2.moveWindow("can", int(width/2), int(height/2)+20)

            key = cv2.waitKey(30)
            if key == 27:
                break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    track(gui=True)

