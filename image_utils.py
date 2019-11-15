from PIL import ImageDraw


def filter_bboxes(prediction, threshold=0.5):
    bboxes = prediction[0]['boxes'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()
    return bboxes[scores > threshold]


def get_pil_image_with_boxes(image, boxes):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, outline=(0, 255, 0))
    return img
