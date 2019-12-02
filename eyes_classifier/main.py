import cv2

from eyes_classifier.models import FullModel
from eyes_classifier.video_capture import run_video_capture

def main(face_detector_path=None, eyes_classifier_path=None):
    model = FullModel(str(face_detector_path), str(eyes_classifier_path))

    cap = cv2.VideoCapture(0)
    run_video_capture(cap, model)