import torch
import cv2
from ssd import build_ssd, SSD
from torchsummary import  summary
import numpy as np
import os

num_classes = 81
image_path = "data/1.jpg"
weights = "weights/ssd300_COCO_140000.pth"
net = build_ssd('test', 300, num_classes)
net.load_state_dict(torch.load(weights))
image_paths = "/home/huaijian/data/coco/coco_val2014"

test = 1
for image in os.listdir(image_paths):
    image_path = image_paths + "/" + str(image)
    #print(image_path)
    test += 1
    if test > 3000:
        break
    if image_path != "/home/huaijian/data/coco/coco_val2014/COCO_val2014_000000542605.jpg":
        continue
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    
    #get the output
    output, output_block = net(image)
    output = output.detach().numpy().tolist()
    #output_block = output_block.numpy().tolist()
    #get the difference, we change the output!!!
    output_new = list()
    output_block_new = list()
#    for i in range(len(output)):
#        for j in range(len(output_block)):
#            if output[i][num_classes + 4] == output_block[j][num_classes + 4]:
#                output[i][num_classes + 4] = -1
#                output_block[j][num_classes + 4] = -1
#    for i in range(len(output)):
#        if output[i][num_classes + 4] != -1:
#            output_new.append(output[i])
#            print("old" + str(i))
#    for i in range(len(output_block)):
#        if output_block[i][num_classes + 4] != -1:
#            output_block_new.append(output_block[i])
#            print("new" + str(i))
#    output = output_new  #attention
#    output_block = output_block_new #attention
    #begin our painting
    if len(output) > 0 or len(output_block) > 0:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))
        for i in range(len(output)): #paint in blue
            print("old" + (str(i) + " " + str(output[i][5]) + " " + str(output[i][0] * 300) + " " + str(output[i][1] * 300) + " " +
                                     str(output[i][2] * 300) + " " + str(output[i][3] * 300))) 
            cv2.rectangle(image, (int(output[i][0] * 300), int(output[i][1] * 300)),\
                 (int(output[i][2] * 300), int(output[i][3] * 300)), (255, 0, 0), 1)
        output = output_block   #attention here !!
#        for i in range(len(output)): #paint in green
#                print("new" + (str(i) + " " + str(output[i][5]) + " " + str(output[i][0]) + " " + str(output[i][1]) + " " +
#                                                               str(output[i][2]) + " " + str(output[i][3])))
#                cv2.rectangle(image, (int(output[i][0] * 300), int(output[i][1] * 300)),\
#                 (int(output[i][2] * 300), int(output[i][3] * 300)), (0, 255, 0), 1)
        cv2.imshow(image_path, image)
        cv2.waitKey()
    else:
        print("same output")
