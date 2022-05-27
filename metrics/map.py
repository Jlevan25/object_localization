import os

import cv2
import numpy as np
from torch import nn
from utils import check_zero_divide

from metrics.metrics import ImageMetric


class ClassAP(ImageMetric):

    def __init__(self, num_classes, iou_threshold: float, confidence_threshold: float,
                 with_cam: bool = False, cam_layer: nn.Module = None, fc_layer: nn.Module = None,
                 save_path: str = None):
        super().__init__()
        self._activations = None
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.curve = -np.ones((num_classes, 11))
        self.iou_threshold = iou_threshold
        self.with_cam = with_cam
        self.save_path = save_path
        if self.with_cam:
            self.cam_layer = cam_layer
            self.fc = fc_layer
            cam_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        activation = output
        self._activations = activation.cpu().data.numpy()

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
            thresh = cv2.threshold(np.uint8(heatmap * 255), 125, 255, cv2.THRESH_OTSU)[1]
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            bboxes = [cv2.boundingRect(c) for c in cnts]
            # confidences = [np.mean(heatmap[x: x + w, y: y + h]) for x, y, w, h in bboxes]
            confidences = [np.mean(heatmap[y: y + h, x: x + w]) for x, y, w, h in bboxes]

        else:
            bboxes = ...
            confidences = ...

        target_bboxes = [tuple(np.round(t['bbox']).astype('int')) for t in target]

        for conf, box in zip(confidences, bboxes):
            x, y, w, h = box
            text = self.class_names[prediction.item()] + ' ' + str(round(conf, 3))
            cv2.rectangle(bb_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.rectangle(bb_img, (x, y), (x + len(text) * 25, y + 30), (0, 255, 255), -1)
            cv2.putText(bb_img, text, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color=(0, 0, 0))

        for box in target_bboxes:
            x, y, w, h = box
            cv2.rectangle(bb_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cv2.imshow('first', bb_img)
        # cv2.imshow('heatmap', cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))
        # cv2.waitKey()

        if self.save_path is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            img = cv2.hconcat([bb_img, heatmap])
            cv2.imwrite(os.path.join(self.save_path, str(target[0]['image_id']) + '.jpg'), img)

        return bboxes, target_bboxes, confidences

    def _iou(self, image, pred_bboxes, target_bboxes):
        predictions = np.array([np.zeros(image.size)] * len(pred_bboxes))
        targets = np.array([np.zeros(image.size)] * len(target_bboxes))

        for i, (x, y, w, h) in enumerate(pred_bboxes):
            predictions[i, x: x + w, y: y + h] = 1

        for i, (x, y, w, h) in enumerate(target_bboxes):
            targets[i, x: x + w, y: y + h] = 1

        ious = []
        for target in targets:
            intersect = (predictions * target).sum((1, 2))
            union = np.maximum(predictions, target).sum((1, 2))
            ious.append(check_zero_divide(intersect, union, val=0))

        ious = np.array(ious)
        zeros = np.zeros_like(ious)
        index = (np.arange(len(ious)), ious.argmax(axis=1)) if ious.shape[-1] > 1 else ious.argmax()
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
        curve[curve < 0] = 0.
        # if precision > self.curve[target, index]:
        smaller_precisions = curve[:index + 1] < precision
        curve[:index + 1][smaller_precisions] = precision
        self.curve[target] = curve

    def get_batch_metric(self, predictions, targets):
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            if prediction == target:
                pred_bb, target_bb, confidences = self._get_bounding_box(prediction, i)

                confidence_bboxes = [bb for bb, confidence in zip(pred_bb, confidences)
                                     if confidence > self.confidence_threshold]
                if len(confidence_bboxes) > 0:
                    precision, recall = self._iou(self.batch_images[i], confidence_bboxes, target_bb)
                    self._change_curve(target, precision, recall)

            # for t in np.arange(0., 1.1, 0.1):
            #     confidence_bboxes = [bb for bb, c in zip(pred_bb, confidences) if c > t]
            #     if len(confidence_bboxes) > 0:
            #         precision, recall = self._iou(self.batch_images[i], confidence_bboxes, target_bb)
            #         self._change_curve(target, precision, recall)

        self._clear_batch()
        return self.curve.mean(1)

    def get_epoch_metric(self):
        return self.curve.mean(1)
