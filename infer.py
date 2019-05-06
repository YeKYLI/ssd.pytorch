import torch
import cv2
from ssd import build_ssd

num_classes = 81
image_path = "data/1.jpg"
weights = "weights/ssd300_COCO_140000.pth"

#def infer()
def get_features_hook(self, input, output):
    print("hooks ", output.data.cpu().numpy().shape)

if __name__ == '__main__':
    net = build_ssd('test', 300, num_classes)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
#load weights to the net
    net.load_state_dict(torch.load(weights))
    output = net(image)
#begin our painting
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    for i in range(len(output)):
        if output[i][5] > 0.3:
          cv2.rectangle(image, (int(output[i][0] * 300), int(output[i][1] * 300)),\
             (int(output[i][2] * 300), int(output[i][3] * 300)), (255, 0, 0), 2)
    cv2.imshow(image_path, image)
    cv2.waitKey()
    print(output.shape)
#get the specific layer value

#    print(net)

