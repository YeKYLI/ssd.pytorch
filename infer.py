import torch
import cv2
from ssd import build_ssd, SSD
from torchsummary import  summary
import numpy as np
import os

num_classes = 81
image_path = "data/1.jpg"
weights = "weights/ssd300_COCO_140000.pth"

#if __name__ == '__main__':

net = build_ssd('test', 300, num_classes)
net.load_state_dict(torch.load(weights))
image_paths = "/home/huaijian/data/coco/coco_val2014"
compare_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
compare_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
test = 1
for image in os.listdir(image_paths):
    image_path = image_paths + "/" + str(image)
    print(image_path) 
    if test > 1000:
        break
#    if image_path != "/home/huaijian/data/coco/coco_val2014/COCO_val2014_000000500116.jpg":
#        continue
    test += 1
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    
    output, output_block = net(image)
    print(output.shape)
    print(output_block.shape)
    com_index = list()
#do the compare
    for i in range(len(output)):
        com_index.append(0)
        for j in range(len(output_block)):
            if output[i][4 + num_classes] == output_block[j][4 + num_classes]:
                com_index[i] = 1

    thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresh_temp1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    thresh_temp2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(output)):
        for j in range(len(thresh)):
            if output[i][5] > thresh[j]:
                thresh_temp1[j] += 1
                if com_index[i] == 1:
                    thresh_temp2[j] += 1
    for i in range(len(thresh)):
        print(str(i) + " " + str(thresh_temp1[i]) + " " +  str(thresh_temp2[i]))
        if thresh_temp1[i] == 0:
            print(0)
        else:
            print(thresh_temp2[i] / thresh_temp1[i])
            compare_score[i] += (thresh_temp2[i] / thresh_temp1[i])
            compare_num[i]  += 1
            print("score" + str(compare_score[i]))
            print("num" + str(compare_num[i]))

#begin our painting
#    output = output.detach().numpy().tolist()
#    image = cv2.imread(image_path)
#    image = cv2.resize(image, (300, 300))
#    for i in range(len(output)):
#        if output[i][5] > 0.5:
#            print(str(i) + " " + str(output[i][5]) + " " + str(output[i][0]) + " " + str(output[i][1])) 
#            cv2.rectangle(image, (int(output[i][0] * 300), int(output[i][1] * 300)),\
#             (int(output[i][2] * 300), int(output[i][3] * 300)), (255, 0, 0), 2)
#    #cv2.imshow(image_path, image)
#    #cv2.waitKey()
#    output = output_block.detach().numpy().tolist()
#    image = cv2.imread(image_path)
#    image = cv2.resize(image, (300, 300))
#    for i in range(len(output)):
#        if output[i][5] > 0.5:
#            print(str(i) + " " + str(output[i][5]) + " " + str(output[i][0]) + " " + str(output[i][1])) 
#            cv2.rectangle(image, (int(output[i][0] * 300), int(output[i][1] * 300)),\
#             (int(output[i][2] * 300), int(output[i][3] * 300)), (255, 0, 0), 2)
#    cv2.imshow(image_path, image)
#    cv2.waitKey()
#    print(output.shape)
#get the specific layer value
#    test_net = build_ssd('train', 300, 81)
#    summary(test_net.cuda(), input_size = (3, 300, 300))
#    print(net)
#do our compare
for x in range(10):
    print(str(x) + " Similarity:  " + str(compare_score[x] / compare_num[x]))
