def bb_intersection_over_union(gt_box, pred_box):
    xA = max(gt_box[0], pred_box[0])
    yA = max(gt_box[1], pred_box[1])
    xB = min(gt_box[2], pred_box[2])
    yB = min(gt_box[3], pred_box[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    gt_boxes_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

    iou = interArea / float(gt_boxes_area + pred_box_area - interArea)

    return iou
