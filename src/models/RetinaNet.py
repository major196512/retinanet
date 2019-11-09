import torch
import torch.nn as nn

from . import ResNet
from . import FPN
from . import InverseFPN
from . import GlobalContext
from . import SubNet

from ..utils.box_utils import BBoxTransform, ClipBoxes
from ..utils.anchors import Anchors
from ..utils.nms_pytorch import nms

from ..losses import FocalLoss

class RetinaNet(nn.Module):
    def __init__(self, num_classes, resnet_size=50, pretrained=True, **kwargs):
        self.num_features = kwargs.pop('num_features', 256)
        self.top_k = kwargs.pop('top_k', 300)
        self.cls_thres = kwargs.pop('cls_thres', 0.05)
        self.inverse = kwargs.pop('inverse', False)
        super(RetinaNet, self).__init__()

        self.ResNet = ResNet.SetUpNet(resnet_size, pretrained)
        resnet_out_size = self.ResNet.get_size()

        if self.inverse:
            self.GlobalFeature = GlobalContext.GlobalNet(resnet_out_size[2], self.num_features)
            self.InverseFPN = InverseFPN.PyramidFeatures(resnet_out_size[0], resnet_out_size[1], resnet_out_size[2])
        else:
            self.FPN = FPN.PyramidFeatures(resnet_out_size[0], resnet_out_size[1], resnet_out_size[2])

        self.SubNet = SubNet.SubNet(num_classes=num_classes, num_features=self.num_features)

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
        if self.inverse:
            G_feature = self.GlobalFeature(C_feature_maps[2])
            P_feature_maps = self.InverseFPN(C_feature_maps, G_feature)
        else:
            P_feature_maps = self.FPN(C_feature_maps)
        classifications, regressions = self.SubNet(P_feature_maps)

        anchors = self.anchors(img_batch)

        if self.training:
            cls_loss, reg_loss = FocalLoss(classifications, regressions, anchors, annotations)
            return cls_loss, reg_loss

        predict_boxes = self.regressBoxes(anchors, regressions)
        predict_boxes = self.clipBoxes(predict_boxes, img_batch)

        predict_boxes = predict_boxes.cpu().detach()[0]
        classifications = classifications.cpu().detach()[0]

        cls_over_thresh = (classifications>self.cls_thres)

        if cls_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return torch.zeros(0).cuda(), torch.zeros(0).cuda(), torch.zeros(0, 4).cuda()

        #scores = []
        scores = torch.zeros(cls_over_thresh.shape)
        #indices = []
        #labels = []

        for c in range(classifications.shape[-1]):
            cls_ind = cls_over_thresh[:, c]
            box = predict_boxes[cls_ind, :]
            score = classifications[cls_ind, c]

            if box.shape[0] == 0: continue

            anchors_nms_idx, max_k = nms(box, score, overlap=0.5, top_k=self.top_k)
            anchors_nms_idx = anchors_nms_idx[:max_k]

            #score = score[anchors_nms_idx]
            #cls_ind = cls_ind.nonzero()[anchors_nms_idx, 0]
            #label = c * torch.ones(cls_ind.shape[0], dtype=torch.int64)
            cls_ind = cls_ind.nonzero().squeeze(dim=1)[anchors_nms_idx]
            scores[cls_ind, c] = score[anchors_nms_idx]

            #scores.append(score)
            #indices.append(cls_ind)
            #labels.append(label)

        #print(scores)
        #scores = torch.cat(scores, dim=0)
        #indices = torch.cat(indices, dim=0)
        #labels = torch.cat(labels, dim=0)
        #print('test')
        scores_max = torch.max(scores, dim=1)[0]
        nms_ind = (scores_max > 0)
        scores = scores[nms_ind]

        labels = scores.argmax(dim=1)
        scores = scores.max(dim=1)[0]
        boxes = predict_boxes[nms_ind]

        #scores_topk = torch.topk(scores, min(self.top_k, scores.shape[0]))[1]
        #print('test')

        #return scores[scores_topk], labels[scores_topk], predict_boxes[0, indices[scores_topk], :]
        return scores.cuda(), labels.cuda(), boxes.cuda()
