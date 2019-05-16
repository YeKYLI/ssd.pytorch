import torch
import numpy as np
from torch.autograd import Function
from ..box_utils import decode
from data import voc as cfg
from .quicksort import quicksort

def box_iou(x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max):
    if x1min > x2max or x1max < x2min:
        return 0
    if y1min > y2max or y1max < y2min:
        return 0
    area_iou = (min(x1max, x2max) - max(x1min, x2min)) * (min(y1max, y2max) - max(y1min, y2min))
    area1 = (x1max - x1min) * (y1max - y1min)
    area2 = (x2max - x2min) * (y2max - y2min)
    return (area_iou / (area1 + area2 - area_iou))

def atom_nms(block_index, all_boxes):
    for i in range(len(block_index)):
        if block_index[i] < 0:
            continue
        for j in range(len(block_index)):
            if j > i and box_iou(all_boxes[block_index[i]][0], all_boxes[block_index[i]][1],
                         all_boxes[block_index[i]][2], all_boxes[block_index[i]][3],
                         all_boxes[block_index[j]][0], all_boxes[block_index[j]][1],
                         all_boxes[block_index[j]][2], all_boxes[block_index[j]][3]) > 0.45:
                #if block_index[j] == 2045:
                #    print("attention !!!!!!!new")
                #    print(block_index[i])
                block_index[j] = -1
    index = list()
    for i in range(len(block_index)):
        if block_index[i] >= 0:
            index.append(block_index[i])
    return index

def block_nms(paper_box, loc, conf, prior_data, num_classes):
    variance = cfg['variance']
    #we get the all predicted boxes and its confidence
    decoded_boxes = decode(loc[0], prior_data, variance)
    conf = conf[0]
    loc = loc[0]
    all_boxes = torch.cat((decoded_boxes, conf), 1)
    indexes = list()
    branch = paper_box[len(paper_box) - 1][0]
    for i in range(branch):
        index = list()
        for j in range(len(paper_box)):
            if(paper_box[j][0] == (i + 1)):
                index.append(j)
        indexes.append(index)

    #in the specific feature map
#    process_boxes = list()
#    for i in range(num_classes):
#        if i > 0:
#            break
#        branch_index = list()
#        for j in range(branch):
#            feature_w = paper_box[indexes[j][len(indexes[j]) - 1]][2] + 1
#            feature_h = paper_box[indexes[j][len(indexes[j]) - 1]][1] + 1
#            index = indexes[j]
#            #threshthod
#            temp_index = list()
#            for k in range(len(index)):
#                if all_boxes[index[k]][i + 5] > 0.1:
#                    temp_index.append(index[k])
#            #print(str(len(index)) + "********1")
#            index = temp_index
#            #print(str(len(index)) + "********2")
#            #quick sort per feature map
#            index_sort = []
#            for k in range(len(index)):
#                index_sort.append([index[k], all_boxes[index[k]][i + 5]])
#            index_sort = quicksort(index_sort, 0, len(index_sort) - 1)
#            for k in range(len(index_sort)):
#                index[k] = index_sort[k][0]
#                #print(str(index[k]) + " " + str(all_boxes[index[k]][i + 5]))
#            #nms per feature map
#            for p in range(len(index)):
#                if index[p] < 0:
#                    continue
#                for q in range(len(index)):
#                    if q > p and box_iou(all_boxes[index[p]][0], all_boxes[index[p]][1],
#                                 all_boxes[index[p]][2], all_boxes[index[p]][3],
#                                 all_boxes[index[q]][0], all_boxes[index[q]][1],
#                                 all_boxes[index[q]][2], all_boxes[index[q]][3])  > 0.45:
#                        index[q] = -1
#            for k in range(len(index)):
#                if index[k] >= 0:
#                    branch_index.append(index[k])
#        #quick sort in all branch
#        index_sort = []
#        index = branch_index
#        for j in range(len(index)):
#            index_sort.append([index[j], all_boxes[index[j]][i + 5]])
#        index_sort = quicksort(index_sort, 0, len(index_sort) - 1)
#        for j in range(len(index_sort)):
#            index[j] = index_sort[j][0]
#        #nms in all branch
#        for j in range(len(index)):
#            if index[j] < 0:
#                continue
#            for k in range(len(index)):
#                if k > j and box_iou(all_boxes[index[j]][0], all_boxes[index[j]][1],
#                                 all_boxes[index[j]][2], all_boxes[index[j]][3],
#                                 all_boxes[index[k]][0], all_boxes[index[k]][1],
#                                 all_boxes[index[k]][2], all_boxes[index[k]][3])  > 0.45:
#                    index[k] = -1
#        for j in range(len(index)):
#            if index[j] >= 0:
#                temp_box = all_boxes[index[j]].detach().numpy().tolist()
#                temp_box.append(index[j])
#                process_boxes.append(temp_box)
#    return process_boxes

            
            
    #in the first, we just test the specific one class !!!
    block_w = 2
    block_h = 2
    final_index = list()
    #per class
    for i in range(num_classes):
        if i != 1:
            continue
        all_branch_index = list()
        #per branch
        for j in range(branch):
            if j != 0:
                break
            feature_w = paper_box[indexes[j][len(indexes[j]) - 1]][2] + 1
            feature_h = paper_box[indexes[j][len(indexes[j]) - 1]][1] + 1
            print("feature map size w = " + str(feature_w) + " h = " +  str(feature_h))
            index = indexes[j]
            #per block
            for p in range(block_h):
                for q in range(block_w):
                    print("index " + str(p * block_w + q))
                    block_index = list()
                    w_low = int(feature_w / block_w) * q
                    w_high = int(feature_w / block_w) * (q + 1)
                    if q == (block_w - 1):
                        w_high = feature_w
                    h_low = int(feature_h / block_h) * p
                    h_high = int(feature_h / block_h) * (p + 1)
                    if p == (block_h - 1):
                        h_high = feature_h
                    #block and threshod
                    for r in range(len(index)):
                        if (paper_box[index[r]][2] >= w_low and paper_box[index[r]][2] < w_high and 
                            paper_box[index[r]][1] >= h_low and paper_box[index[r]][1] < h_high and all_boxes[index[r]][i + 4] > 0.1):
                            block_index.append(index[r])
                    #sort per block
                    index_sort = []
                    for r in range(len(block_index)):
                        index_sort.append([block_index[r], all_boxes[block_index[r]][i + 4]])
                    index_sort = quicksort(index_sort, 0, len(index_sort) - 1)
                    for r in range(len(block_index)):
                        block_index[r] = index_sort[r][0]
                        print("new " + str(block_index[r]) + " " + str(all_boxes[block_index[r]][i + 4]))
                    #nms per block
                    block_index = atom_nms(block_index, all_boxes)
                    for x in range(len(block_index)):
                        all_branch_index.append(block_index[x])
        #sort and nms in all feature map
        index_sort = []
        for x in range(len(all_branch_index)):
            index_sort.append([all_branch_index[x], all_boxes[all_branch_index[x]][i + 4]])
        index_sort = quicksort(index_sort, 0, len(index_sort) - 1)
        for x in range(len(all_branch_index)):
            all_branch_index[x] = index_sort[x][0]
        final_index = atom_nms(all_branch_index, all_boxes)
        per_boxes = list()
        for x in range(len(final_index)):
            temp_box = all_boxes[final_index[x]].detach().numpy().tolist()
            print("new" + str(x) + " " + str(final_index[x]) + " " + str(all_boxes[final_index[x]][i + 4]))
            temp_box.append(final_index[x])
            per_boxes.append(temp_box)
        return per_boxes 

    
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
        all_index = []
        process_boxes = []
        #we get the exact index
        for i in range(self.num_classes - 1):
            if i > 0:
                break
            #threshthod
            index = list()
            for j in range(len(all_boxes)):
                if all_boxes[j][i + 5] > 0.1 and j < 5777: #attention !!!
                    index.append(j)
            #quick sort
            index_sort = []
            for j in range(len(index)):
                index_sort.append([index[j], all_boxes[index[j]][i + 5]])
            index_sort = quicksort(index_sort, 0, len(index_sort) - 1)
            for j in range(len(index_sort)):
                index[j] = index_sort[j][0]
            #do nms in specific class
            for j in range(len(index)):
                if index[j] < 0:
                    continue
                for k in range(len(index)):
                    if k > j and box_iou(all_boxes[index[j]][0], all_boxes[index[j]][1],
                                 all_boxes[index[j]][2], all_boxes[index[j]][3],
                                 all_boxes[index[k]][0], all_boxes[index[k]][1],
                                 all_boxes[index[k]][2], all_boxes[index[k]][3])  > 0.45:
                        #if index[k] == 2055:
                        #    print(index[j])
                        #    print("attention !!! old")
                        index[k] = -1
            count = 0
            #attention !!!!!!!!!!!!!!!
            #index.clear()
            #index.append(2045)
            #index.append(2051)
            #index.append(2055)
            #index.append(2201)
            for j in range(len(index)):
                if index[j] >= 0:
                    temp_box = all_boxes[index[j]].numpy().tolist()
                    temp_box.append(index[j])
                    process_boxes.append(temp_box)
                    print("old" + str(count) +  " " + str(index[j]) + " " + str(all_boxes[index[j]][i + 5]))
                    count = count + 1
        process_boxes = torch.FloatTensor(process_boxes)
        return process_boxes


