import argparse
import os

import torch
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F
from hub_classifier import resnet
from hub_classifier.resnet import resnet50,resnet34
#from resnet import resnet50, resnet34
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Hub:

    # def __init__(self, model_path):
    #
    #     if torch.cuda.is_available():
    #         se
    #         self.model = torch.load(model_path, map_location='cpu')
    #     self.model.eval()
    #     self.transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((224, 224)),  # 将图像转化为128 * 128
    #         transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    #         transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 归一化
    #     ])

    def __init__(self, weights_path=None, device="cuda"):

        self._weights_path = weights_path
        self._device = device

        #self.net = UNet(self.N_CLS).to('cpu')

        self.net = resnet50(num_classes=3)

        if weights_path is not None:
            self.load_weights(self._weights_path, self._device)

        # self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 将图像转化为128 * 128
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 归一化
        ])


        # if torch.cuda.is_available():
        #     self.model = torch.load(model_path).to(device)
        # else:
        #     self.model = torch.load(model_path, map_location='cpu')
        # self.model.eval()
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),  # 将图像转化为128 * 128
        #     transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        #     transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 归一化
        # ])
    def load_weights(self, weights_path, device="cuda"):
        self._weights_path = weights_path
        self._device = torch.device(self._device if torch.cuda.is_available() else "cpu")

        self.net.load_state_dict(torch.load(self._weights_path, map_location='cpu'))
        self.net.to(self._device)
        self.net.eval()

    def detect(self, image):

        image = self.transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        #outputs = self.model(image)
        outputs = self.net(image)
        prob = F.softmax(outputs, dim=1)
        pred = torch.argmax(prob, dim=1)
        #pred = pred.numpy()
        pred = pred.cpu().numpy()
        return pred[0]

def main():

    print(os.getcwd())
    hub = Hub()
    hub.load_weights("./model/arrow_hub_other_-11.pth", "cuda")
    #hub = Hub('./model/hub-17.pth')
    #root = './data/test/Sedan'
    root = r'E:\Winddet\classfy_net\hub_classifier\data\train\arrow'
    #img_list = [f for f in os.listdir(root) if f.endswith('.jpg')]
    img_list = [f for f in os.listdir(root) if f.endswith('.jpg')]
    a = 0
    for img in img_list:
        image = cv2.imread(os.path.join(root, img))
        print(img)

        pred = hub.detect(image)
        # if pred == 0:
        #     print('Sedan')
        # else:
        #     print('SUV')
        if pred == 0:
            print('other')
            # cv2.imwrite(r'E:\Winddet\classfy_net\hub_classifier\data\train\arrow'+'\\'+'s--50'+img,image)


        elif(pred == 1):
            print('arrow')
            a += 1
        elif(pred==2):
            print('hub')

        else:
            print('__')
    print(a/5883)
        #cv2.imshow('test', image)
        #cv2.waitKey(0)

if __name__ == '__main__':
    main()
