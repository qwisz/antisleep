import cv2

from eyes_classifier.models import FullModel
from eyes_classifier.video_capture import run_video_capture, process_video

def main(face_detector_path=None, eyes_classifier_path=None, path_to_video=None, output_path='output.avi'):
    model = FullModel(str(face_detector_path), str(eyes_classifier_path))

    cap = cv2.VideoCapture(path_to_video)
    process_video(cap, model, output_path)
    # run_video_capture(cap, model)