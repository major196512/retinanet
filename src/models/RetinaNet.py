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

        scores = torch.max(classifications, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores>0.05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classifications = classifications[:, scores_over_thresh, :]
        predict_boxes = predict_boxes[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx, max_k = nms(torch.cat([predict_boxes, scores], dim=2)[0, :, :], 0.5)
        anchors_nms_idx = anchors_nms_idx[:max_k]

        nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)

        return [nms_scores, nms_class, predict_boxes[0, anchors_nms_idx, :]]
