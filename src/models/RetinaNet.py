import torch
import torch.nn as nn

from models import ResNet
from models import FPN
from models import SubNet

from utils.box_utils import BBoxTransform, ClipBoxes
from utils.anchors import Anchors
from utils.nms_pytorch import nms

from losses import FocalLoss

class RetinaNet(nn.Module):
    def __init__(self, num_classes, resnet_size=50, pretrained=True, num_features=256):
        super(RetinaNet, self).__init__()

        self.ResNet = ResNet.SetUpNet(resnet_size, pretrained)
        resnet_out_size = self.ResNet.get_size()
        self.FPN = FPN.PyramidFeatures(resnet_out_size[0], resnet_out_size[1], resnet_out_size[2])
        self.SubNet = SubNet.SubNet(num_classes=num_classes, num_features=num_features)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.freeze_bn()

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        C_feature_maps = self.ResNet(img_batch)
        P_feature_maps = self.FPN(C_feature_maps)
        classifications, regressions = self.SubNet(P_feature_maps)

        anchors = self.anchors(img_batch)

        if self.training:
            return FocalLoss(classifications, regressions, anchors, annotations)

        predict_boxes = self.regressBoxes(anchors, regressions)
        predict_boxes = self.clipBoxes(predict_boxes, img_batch)

        predict_boxes = predict_boxes.cpu().detach()
        classifications = classifications.cpu().detach()

        cls_over_thresh = (classifications>0.05)[0, :, :]

        if cls_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        scores = []
        indices = []
        labels = []

        for c in range(classifications.shape[2]):
            cls_ind = cls_over_thresh[:, c]
            box = predict_boxes[0, cls_ind, :]
            score = classifications[0, cls_ind, c]

            if box.shape[0] == 0: continue

            anchors_nms_idx, max_k = nms(box, score, 0.5)
            anchors_nms_idx = anchors_nms_idx[:max_k]

            score = score[anchors_nms_idx]
            cls_ind = cls_ind.nonzero()[anchors_nms_idx, 0]
            label = c * torch.ones(cls_ind.shape[0], dtype=torch.int64)

            scores.append(score)
            indices.append(cls_ind)
            labels.append(label)

        scores = torch.cat(scores, dim=0)
        indices = torch.cat(indices, dim=0)
        labels = torch.cat(labels, dim=0)

        scores_topk = torch.topk(scores, min(200, scores.shape[0]))[1]

        return [scores[scores_topk], labels[scores_topk], predict_boxes[0, indices[scores_topk], :]]
