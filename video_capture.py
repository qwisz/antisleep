import cv2 as cv
import numpy as np


def draw_bbox(frame: np.array, face_box: tuple, eyes_status: int):
    xmin, ymin, xmax, ymax = face_box
    top_left = (xmin, ymin)
    bottom_right = (xmax, ymax)
    cv.rectangle(frame,
                 top_left,
                 bottom_right,
                 (0, 155, 255),
                 2
                 )
    text_top_left = (int(xmin), int(ymin) - 3 )
    result = 'closed' if eyes_status == 1 else 'opened'
    cv.putText(frame, result, text_top_left, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv.LINE_AA)
    return frame


def run_video_capture(cap, model):
    while True:
        try:
            ret, frame = cap.read()
            eyes_status, boxes = model(frame)
            if eyes_status is not None and boxes is not None:
                for i, (eyes, box) in enumerate(zip(eyes_status, boxes)):
                    frame = draw_bbox(frame, box, eyes)

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            print("Releasing")
            break

    cap.release()
    cv.destroyAllWindows()
