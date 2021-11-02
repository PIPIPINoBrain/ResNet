import json
import os
import math
import cv2
import numpy as np
from PIL import Image

path_label = r'./label_json'
path_in = r'./ori_img'
path_out = r'./img_json'

#将带有json标注的图片文件找出来并保存
def get_imgjson(path_label, path_in, path_out):
    namelist = os.listdir(path_label)
    for name in namelist:
        basename = name.split('.')[0]
        print("{}已复制".format(name))
        img = cv2.imread(path_in+'\\'+basename+'.JPG', 1)
        #img = Image.open(path_in+'\\'+basename+'.JPG')
        cv2.imwrite(path_out+'\\'+basename+'.JPG', img)
        print("{}已复制".format(name))
    return 0


#将带有Json标注的图片绕（标注点）旋转并截取箭头图片、、先旋转再截取，无黑边
def get_orijson(path_label, path_in, path_out, theta):
    namelist = os.listdir(path_label)
    for name in namelist:
        with open(path_label+'\\'+name, 'r', encoding='utf-8') as jf:
            label_inside = json.load(jf)
            basename = name.split('.')[0]
            print(label_inside['shapes'][0]['points'][0])   #标注点信息


            x_c = (label_inside['shapes'][0]['points'])[0][0] #旋转中心
            y_c = (label_inside['shapes'][0]['points'])[0][1]
            img = cv2.imread(path_in+'\\' + basename + '.JPG', 1)
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D(center=(x_c,y_c), angle=theta, scale=1)
            rot_img = cv2.warpAffine(img, M, (cols,rows))

            lenth = 256
            img_ = rot_img[int(y_c-lenth):int(y_c+lenth),int(x_c-lenth):int(x_c+lenth)]          #截取范围大小
            print(img_.shape)
            if(y_c-lenth<=0 or y_c+lenth>=rows or x_c-lenth<=0 or x_c+lenth>=cols):
                print("超出截取范围")
            else:
                cv2.imwrite(path_out+'\\'+str(theta)+'-' + basename+'.jpg', img_)
        print("{}处理完成".format(name))
    return 0

#先截图再旋转，有黑边
def get_orijson_(path_label, path_in, path_out, theta):
    namelist = os.listdir(path_label)
    for name in namelist:
        with open(path_label+'\\'+name, 'r', encoding='utf-8') as jf:
            label_inside = json.load(jf)
            basename = name.split('.')[0]
            print(label_inside['shapes'][0]['points'][0])   #标注点信息
            x_c = (label_inside['shapes'][0]['points'])[0][0] #旋转中心
            y_c = (label_inside['shapes'][0]['points'])[0][1]

            img = cv2.imread(path_in+'\\' + basename + '.JPG', 1)
            rows, cols = img.shape[:2]
            k = 256
            img_ = img[int(y_c - k):int(y_c + k), int(x_c - k):int(x_c + k)]  # 截取范围大小
            print(img_.shape)
            if (y_c - k <= 0 or y_c + k >= rows or x_c - k <= 0 or x_c + k >= cols):
                print("超出截取范围")
            else:
                row, col = img_.shape[:2]
                x_ = col/2
                y_ = row/2
                print(x_,y_)
                M = cv2.getRotationMatrix2D(center=(x_,y_), angle=theta, scale=1)
                rot_img = cv2.warpAffine(img_, M, (512, 512))
                #rot_img = cv2.resize(rot_img,(400,400))

            cv2.imwrite(path_out+'\\'+str(theta)+'-' + basename+'.jpg', rot_img)
        print("{}处理完成".format(name))
    return 0



#普通图片的旋转保存
def get_rot(path_in, path_out,theta):
    namelist = os.listdir(path_in)
    for name in namelist:
        img = cv2.imread(path_in + '\\' + name, 1)
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D(center=(cols/2, rows/2), angle=theta, scale=1)
        rot_img = cv2.warpAffine(img, M, (cols, rows))
        basename = name.split('.')[0]
        cv2.imwrite(path_out + '\\' + str(theta) + '-' + basename + '.jpg', rot_img)
    return 0

#按照json标注点旋转，不截取
def get_rotjson(path_img, path_json, path_out, theta):
    namelist = os.listdir(path_json)
    for name in namelist:
        with open(path_json+'\\'+name,'r') as tf:
            basename = name.split('.')[0]
            label_inside = json.load(tf)
            print(label_inside['shapes'][0]['points'][0])   #标注点信息
            x_c = (label_inside['shapes'][0]['points'])[0][0] #旋转中心
            y_c = (label_inside['shapes'][0]['points'])[0][1]
            img = cv2.imread(path_img + '\\' + basename + '.jpg', 1)
            print(1)
            rows, cols = img.shape[:2]
            if(theta==0):

                rot_img_2 = cv2.resize(img,(rows,int(1.5*cols)))


                rot_img_1 = cv2.resize(img,(int(1.5*rows),cols))

                rot_img_0 = cv2.resize(img,(rows, cols))
                cv2.imwrite(path_out + '\\' + str(theta)+ '1.5col' + basename+'.jpg', rot_img_2)
                cv2.imwrite(path_out + '\\' + str(theta)+ '1.5row' + basename+'.jpg', rot_img_1)
                cv2.imwrite(path_out + '\\' + str(theta) + '1.1cc' +basename+'.jpg', rot_img_0)
            else:
                M = cv2.getRotationMatrix2D(center=(x_c, y_c), angle=theta, scale=1)
                rot_img = cv2.warpAffine(img, M, (rows, int(1.5 * cols)))
                cv2.imwrite(path_out + '\\' + str(theta) + basename + '.jpg', rot_img)
                rot_img_0 = cv2.flip(rot_img,0)
                rot_img_1 = cv2.flip(rot_img,1)
                cv2.imwrite(path_out + '\\' + str(theta)+ 'flip0' + basename + '.jpg', rot_img_0)
                cv2.imwrite(path_out + '\\' + str(theta) + 'flip1' + basename + '.jpg', rot_img_0)
    return 0





path_label = r'./label_json'
path_ori = r'./ori_img'
path_imgj = r'./img_json'
path_img = r'./arrowhead/img_arrow'
path_out = r'./arrowhead/img_other'
# get_imgjson(path_label, path_ori, path_imgj)
for i in range(0,359,30):
    # get_orijson_(path_label, path_imgj, path_img, i)
    get_rot(r'E:\Winddet\classfy_net\hub', r'E:\Winddet\classfy_net\hub_classifier\data\train\hub', i)



# path_img = r'E:\Winddet\classfy_net\imageori\img'
# path_json = r'E:\Winddet\classfy_net\imageori\label'
# path_out = r'E:\Winddet\classfy_net\imageori\out'
# for i in range(0,359,60):
#     get_rotjson(path_img, path_json, path_out, i)