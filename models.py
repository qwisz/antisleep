import cv2 as cv
import torch
from torchvision.models import mobilenet_v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor

from image_utils import crop_image


class FullModel:
    def __init__(self, face_detector_path, eyes_classidier_path):
        self.face_detector = FaceDetector(face_detector_path)
        self.eyes_classifier = EyesClassifier(eyes_classidier_path)

    def __call__(self, image, face_threshold=0.7):
        boxes, scores = self.face_detector(image)

        # face = image.crop(*boxes.numpy())  # if several faces?
        boxes = boxes.numpy()
        if len(boxes) == 0:
            return None, None

        # todo: finalize for several persons
        if len(boxes) > 1:
            return None, None

        face = crop_image(image, *boxes) # if several faces?

        is_close = self.eyes_classifier(face).numpy()

        return is_close, boxes


class FaceDetector:
    def __init__(self, model_path):
        num_classes = 2
        self.to_tensor = ToTensor()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = fasterrcnn_resnet50_fpn()
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __call__(self, image, threshold=0.7):
        image = self.to_tensor(image)
        image = image.to(self.device)
        prediction = self.model([image])

        bboxes = prediction[0]['boxes']
        scores = prediction[0]['scores']

        to_keep = scores > threshold
        bboxes = bboxes[to_keep]
        scores = scores[to_keep]

        return bboxes.cpu().detach(), scores.cpu().detach()  # detach().numpy() ??


class EyesClassifier:
    def __init__(self, model_path):
        num_classes = 2
        self.to_tensor = ToTensor()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = mobilenet_v2()
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __call__(self, image):
        # image = image.resize((224, 224))
        image = cv.resize(image, (224, 224))
        image = self.to_tensor(image).unsqueeze(0)
        image = image.to(self.device)

        outputs = self.model(image)

        _, prediction = torch.max(outputs, 1)

        return prediction.cpu().detach()
