import cv2
import numpy as np
import torch
from torch import nn

from metrics.metrics import ImageMetric


class mAP(ImageMetric):

    def __init__(self, classes, iou_threshold: float, with_cam: bool = False, cam_layer: nn.Module = None,
                 fc_layer: nn.Module = None):
        super().__init__()
        self._activations = None
        self.classes = classes
        self.curve = np.zeros((classes, 11))
        self.iou_threshold = iou_threshold
        self.with_cam = with_cam
        if self.with_cam:
            self.cam_layer = cam_layer
            self.fc = fc_layer
            cam_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        activation = output
        self._activations = activation.cpu().data.numpy()

    @staticmethod
    def show_cam_on_image(img, heatmap):
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # cam = heatmap + img
        # cam = cam / np.max(cam)
        cv2.imshow('heatmap', heatmap)
        # cv2.waitKey()

    def _get_cam(self, prediction, activation):
        weights = self.fc.weight[prediction].cpu().data.numpy()
        mul_shape = np.ones(len(activation.shape), dtype='int')
        mul_shape[0] = weights.shape[0]
        return np.sum(weights.reshape(mul_shape) * activation, axis=0)

    @staticmethod
    def _get_heatmap(cam, image):
        heatmap = np.maximum(cam, 0)
        heatmap = cv2.resize(cam, image.size)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        return heatmap

    def _get_bounding_box(self, prediction, index):
        image, target = self.batch_images[index], self.batch_targets[index]
        bb_img = cv2.merge(cv2.split(np.asarray(image))[::-1])

        if self.with_cam:
            cam = self._get_cam(prediction, self._activations[index])
            heatmap = self._get_heatmap(cam, image)
            thresh = cv2.threshold(np.uint8(heatmap * 255), 0, 255, cv2.THRESH_OTSU)[1]
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            bboxes = [cv2.boundingRect(c) for c in cnts]
            confidences = [np.mean(heatmap[y: y + h, x: x + w]) for x, y, w, h in bboxes]
        else:
            bboxes = ...
            confidences = ...

        target_bboxes = [tuple(np.round(t['bbox']).astype('int')) for t in target]

        for conf, box in zip(confidences, bboxes):
            x, y, w, h = box
            cv2.rectangle(bb_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(bb_img, str(conf), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0, 0, 255))

        for box in target_bboxes:
            x, y, w, h = box
            cv2.rectangle(bb_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cv2.imshow('first', bb_img)
        # cv2.waitKey()

        return bboxes, target_bboxes, confidences

    def _iou(self, image, pred_bboxes, target_bboxes):
        predictions = np.array([np.zeros(image.size)] * len(pred_bboxes))
        targets = np.array([np.zeros(image.size)] * len(target_bboxes))
        for i, (x, y, w, h) in enumerate(pred_bboxes):
            predictions[i, y: y + h, x: x + w] = 1

        for i, (x, y, w, h) in enumerate(target_bboxes):
            targets[i, y: y + h, x: x + w] = 1

        ious = []
        for target in targets:
            intersect = (predictions * target).sum((1, 2))
            union = np.maximum(predictions, target).sum((1, 2))
            ious.append(intersect / union if np.all(union != 0) else 0)

        ious = np.array(ious)
        if ious.shape[-1] > 1:
            zeros = np.zeros_like(ious)
            index = np.arange(len(ious)), ious.argmax(axis=1)
            zeros[index] = ious[index]
            ious = zeros

        preds = np.sum(ious > self.iou_threshold, 1)
        tp = np.sum(preds > 0)
        fp = ious.shape[1] - tp
        fn = np.sum(preds == 0)

        precision = np.round(tp / (tp + fp + 1e-6), 3)
        recall = np.round(tp / (tp + fn + 1e-6), 1)

        return precision, recall

    def _change_curve(self, target, precision, recall):
        index = int(recall * 10)
        curve = self.curve[target]
        if precision > curve[index]:
            self.curve[target, :index] = precision

    def get_batch_metric(self, predictions, targets):
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            pred_bb, target_bb, confidences = self._get_bounding_box(prediction, i)
            for t in np.arange(0., 1.1, 0.1):
                confidence_bboxes = [bb for bb, c in zip(pred_bb, confidences) if c > t]
                if len(confidence_bboxes) > 0:
                    precision, recall = self._iou(self.batch_images[i], confidence_bboxes, target_bb)
                    self._change_curve(target, precision, recall)

        self._clear_batch()
        return self.curve.mean(1)/self.classes

    def get_epoch_metric(self):
        return self.curve.mean(1)/self.classes
