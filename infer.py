import torch
import cv2
from ssd import build_ssd, SSD
from torchsummary import  summary
import numpy as np
#from tensorboardX import SummaryWriter
#writer = SummaryWriter('log')
#dummy_input = torch.rand(20, 3, 300, 300)
#test_net = build_ssd('train', 300, 81)
#with SummaryWriter(comment = "ssd") as w:
#    w.add_graph(test_net, dummy_input)


num_classes = 81
image_path = "data/1.jpg"
weights = "weights/ssd300_COCO_140000.pth"

if __name__ == '__main__':
    net = build_ssd('test', 300, num_classes)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
#load weights to the net
    net.load_state_dict(torch.load(weights))
    output, output_block = net(image)
    output = output_block
#begin our painting
    output = output.detach().numpy().tolist()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    for i in range(len(output)):
        if output[i][5] > 0.3:
            print(str(i) + " " + str(output[i][5]) + " " + str(output[i][0]) + " " + str(output[i][1])) 
            cv2.rectangle(image, (int(output[i][0] * 300), int(output[i][1] * 300)),\
             (int(output[i][2] * 300), int(output[i][3] * 300)), (255, 0, 0), 2)
    cv2.imshow(image_path, image)
    cv2.waitKey()
#    print(output.shape)
#get the specific layer value
#    test_net = build_ssd('train', 300, 81)
#    summary(test_net.cuda(), input_size = (3, 300, 300))
#    print(net)

