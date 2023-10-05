import xml.etree.ElementTree as ET
 
import pickle
import os
from os import listdir , getcwd
from os.path import join
import glob
import argparse


# classes = ["cone tank", "water horse bucket"]
classes = ["dog", "cat"]
 
def convert(size, box):
 
    dw = 1.0/size[0] #计算在0到1之间的这个图片尺寸区域
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_name,label_path,label_xml_path):
    # in_file = open('./indata/'+image_name[:-3]+'xml') #xml文件路径
    # out_file = open('./labels/train/'+image_name[:-3]+'txt', 'w') #转换后的txt文件存放路径
    # out_file = open('F:/AI/Python/yolov5/MyDatasets/labels/train/'+image_name[:-3]+'txt', 'w') #转换后的txt文件存放路径
    out_file = open(label_path+image_name[:-3]+'txt', 'w')
    # f = open('./indata/'+image_name[:-3]+'xml')
    # f = open('F:/AI/Python/yolov5/MyDatasets/data_save/'+image_name[:-3]+'xml')
    f = open(label_xml_path+image_name[:-3]+'xml')
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'): #循环获取目标
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()


# 设置参数
def parse_opt():
    parser = argparse.ArgumentParser() #argparse.ArgumentParser()建立解析对象
    #add_argument增加属性
    parser.add_argument("--train_images_path_jpg", type = str, default="F:/AI/Python/yolov5_rewrite/MyDatasets/images/train/*.jpg")
    parser.add_argument("--train_labels_path", type=str,default="F:/AI/Python/yolov5_rewrite/MyDatasets/labels/train/")
    parser.add_argument("--train_labels_xml_path", type = str ,default= "F:/AI/Python/yolov5_rewrite/MyDatasets/labels_xml/")
    return parser.parse_args()# 调用parse_args()方法进行解析
 
if __name__ == '__main__':
    a = parse_opt()
    # print(a.train_images_path_jpg)
    # exit()
    #glob模块是最简单的模块之一，内容非常少。用它可以查找符合特定规则的文件路径名。
    # print(glob.glob(a.train_images_path_jpg))
    # exit()
    for image_path in glob.glob(a.train_images_path_jpg): #每一张图片都对应一个xml文件这里写xml对应的图片的路径
        image_name = image_path.split('\\')[-1]
        convert_annotation(image_name,a.train_labels_path,a.train_labels_xml_path)

    
    #设置参数



    