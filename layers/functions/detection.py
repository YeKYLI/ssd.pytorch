import torch
import numpy as np
from torch.autograd import Function
from ..box_utils import decode
from data import voc as cfg
from .quicksort import quicksort

class Paper_box():
    def data(self, index, x, y, xmin, ymin, xmax, ymax, confidences):
        self.index = index
        self.x = x
        self.y = y
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidences = confidences

def box_iou(x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max):
    if x1min > x2max or x1max < x2min:
        return 0
    if y1min > y2max or y1max < y2min:
        return 0
    area_iou = (min(x1max, x2max) - max(x1min, x2min)) * (min(y1max, y2max) - max(y1min, y2min))
    area1 = (x1max - x1min) * (y1max - y1min)
    area2 = (x2max - x2min) * (y2max - y2min)
    return (area_iou / (area1 + area2 - area_iou))

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        #we get the all predicted boxes and its confidence
        decoded_boxes = decode(loc_data[0], prior_data, self.variance)
        conf_data = conf_data[0]
        loc_data = loc_data[0]
        all_boxes = torch.cat((decoded_boxes, conf_data), 1)
        #all_boxes = all_boxes[:300, :]
        print(str(len(all_boxes)) + "***************")
        index = []
        #ka yi ge yu  zhi  lai  jia  su
        #we get the exact index
        for i in range(self.num_classes - 1):
            if i > 0:
                break
            index.clear()
            for j in range(len(all_boxes)):
                index.append(j)
            #bubble sort
#            for j in range(len(all_boxes)):
#                for k in range(len(all_boxes)):
#                    if k > j and all_boxes[index[j]][i + 5] < all_boxes[index[k]][i + 5]:
#                        index[j] = index[j] + index[k]
#                        index[k] = index[j] - index[k]
#                        index[j] = index[j] - index[k]
            #quick sort
            index_sort = []
            for j in range(len(all_boxes)):
                index_sort.append([j, all_boxes[j][i + 5]])
            index_sort = quicksort(index_sort, 0, len(index_sort) - 1)
            index.clear()
            #threshthod
            for j in range(len(index_sort)):
                if all_boxes[index_sort[j][0]][i + 5] > 0.2:
                    index.append(index_sort[j][0])
            #do nms in specific class
            for j in range(len(index)):
                if index[j] < 0:
                    continue
                for k in range(len(index)):
                    if k > j and box_iou(all_boxes[index[j]][0], all_boxes[index[j]][1],
                                 all_boxes[index[j]][2], all_boxes[index[j]][3],
                                 all_boxes[index[k]][0], all_boxes[index[k]][1],
                                 all_boxes[index[k]][2], all_boxes[index[k]][3])  > 0.45:
                        index[k] = -1
        process_boxes = []
        for i in range(len(index)):
            if index[i] >= 0:
                process_boxes.append(all_boxes[index[i]].numpy().tolist())
        process_boxes = torch.FloatTensor(process_boxes)
        return process_boxes

        # Decode predictions into bboxes.
#        for i in range(num):
#            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
#            # For each class, perform nms
#            conf_scores = conf_preds[i].clone()
#
#            for cl in range(1, self.num_classes):
#                c_mask = conf_scores[cl].gt(self.conf_thresh)
#                scores = conf_scores[cl][c_mask]
#                if scores.size(0) == 0:
#                    continue
#                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
#                boxes = decoded_boxes[l_mask].view(-1, 4)
#                # idx of highest scoring and non-overlapping boxes per class
#                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
#                output[i, cl, :count] = \
#                    torch.cat((scores[ids[:count]].unsqueeze(1),
#                               boxes[ids[:count]]), 1)
#        flt = output.contiguous().view(num, -1, 5)
#        _, idx = flt[:, :, 0].sort(1, descending=True)
#        _, rank = idx.sort(1)
#        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
#        return output

