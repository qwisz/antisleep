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
    text_top_left = (int(xmin), int(ymin) - 3)
    result = 'closed' if eyes_status == 1 else 'opened'
    cv.putText(frame, result, text_top_left, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 2, cv.LINE_AA)
    return frame


def run_video_capture(cap, model):
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                eyes_status, boxes = model(frame_rgb)
                if eyes_status is not None and boxes is not None:
                    for i, (eyes, box) in enumerate(zip(eyes_status, boxes)):
                        frame = draw_bbox(frame, box, eyes)

                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        except KeyboardInterrupt:
            print("Releasing")
            break

    cap.release()
    cv.destroyAllWindows()


def process_video(vid, model, output_path):
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, 15.0, size)
    while vid.isOpened():
        try:
            ret, frame = vid.read()

            if ret:
                eyes_status, boxes = model(frame)
                if eyes_status is not None and boxes is not None:
                    for i, (eyes, box) in enumerate(zip(eyes_status, boxes)):
                        frame = draw_bbox(frame, box, eyes)

                out.write(frame)

            else:
                break


        except KeyboardInterrupt:
            print("Releasing")
            break
    return out
