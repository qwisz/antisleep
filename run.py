from eyes_classifier.main import main

face_detector_path = 'model_weights/face_detector_opencv.pt'
eyes_classifier_path = 'model_weights/eyes_mobilenet_opencv.pt'

main(face_detector_path, eyes_classifier_path)