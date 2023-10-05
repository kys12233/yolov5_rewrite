import xml.etree.ElementTree as ET
import glob
import argparse



classes = ["dog", "cat"]


#传入图片的尺寸size = (w,h)，框bbox=(xmin,xmax,ymin,ymax)（框只有单个）
def convert(size,bbox):
    dx = 1.0 / size[0]
    dy = 1.0 / size[1]

    # 计算在传入的尺寸上的中心点坐标和宽高
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]

    #计算在0-1之间的中心点坐标和宽高，要将结果转化成0-1之间的一个原因是防止训练过程中梯度溢出
    x = x * dx
    y = y * dy
    w = w * dx
    h = h * dy

    return (x,y,w,h)


# 根据获取的图片的名称进行对应替换成对应的经过标记的得到保存的xml文件，并解析xml文件后，将标签转化为0-1之间并进行保存
#convert annotation 的中文意思是：转换 注释
def convert_annotation(image_name,label_path,label_xml_path):

    out_flie_txt = open(label_path + image_name[:-3] + 'txt', "w")

    #读取xml文件
    xml_file = open(label_xml_path + image_name[:-3] + 'xml') #直接使用open打开xml文件
    xml_text = xml_file.read()  # xml文件读取
    root = ET.fromstring(xml_text) #将xml文件读取后的结果从string转换成root的对象
    xml_file.close() #关闭xml文件

    #解析root
    size = root.find("size")
    w = int(size.find('width').text)
    h = int(size.find("height").text)

    for object in root.iter("object"):
        name = object.find("name").text #.text指的是属性，如果不加.text，返回的是一个object.find("name")对象，
        #打印出来会显示<Element 'name' at xxx> :xxx表示地址
        if name not in classes:
            print("classes错误")
            continue
        
        #由于实际运算的时候，需要将类别转换成数字，所以采用的方式是使用一个列表classes，然后使用index的方法进行
        #获取索引，使用索引作为标签
        name_id = classes.index(name)
        
        bndbox = object.find("bndbox")
        xmin = float(bndbox.find('xmin').text)
        # print(xmin)
        ymin = float(bndbox.find('ymin').text)
        # print(ymin)
        xmax = float(bndbox.find('xmax').text)
        # print(xmax)
        ymax = float(bndbox.find("ymax").text)
        # print(ymax)

        b = [xmin,xmax,ymin,ymax]
        bbox = convert([w,h],b)
        out_flie_txt.write(str(name_id) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + '\n')



def parse_opt():
    parser = argparse.ArgumentParser() #argparse.ArgumentParser()建立解析对象
    #add_argument增加属性
    parser.add_argument("--train_images_path_jpg", type = str, default="F:/AI/Python/yolov5_rewrite/MyDatasets/images/train/*.jpg")
    parser.add_argument("--train_labels_path", type=str,default="F:/AI/Python/yolov5_rewrite/MyDatasets/labels/train/")
    parser.add_argument("--train_labels_xml_path", type = str ,default= "F:/AI/Python/yolov5_rewrite/MyDatasets/labels_xml/")
    parser.add_argument("--val_images_path_jpg", type = str, default="F:/AI/Python/yolov5_rewrite/MyDatasets/images/val/*.jpg")
    parser.add_argument("--val_labels_path", type=str,default="F:/AI/Python/yolov5_rewrite/MyDatasets/labels/val/")
    parser.add_argument("--val_labels_xml_path", type = str ,default= "F:/AI/Python/yolov5_rewrite/MyDatasets/labels_xml/")
    return parser.parse_args()# 调用parse_args()方法进行解析
 
if __name__ == '__main__':
    a = parse_opt()
    # print(a.train_images_path_jpg)
    # exit()
    #glob模块是最简单的模块之一，内容非常少。用它可以查找符合特定规则的文件路径名。
    # print(glob.glob(a.train_images_path_jpg))
    # exit()
    # 训练集的处理
    for image_path in glob.glob(a.train_images_path_jpg): #每一张图片都对应一个xml文件这里写xml对应的图片的路径
        image_name = image_path.split('\\')[-1]
        convert_annotation(image_name,a.train_labels_path,a.train_labels_xml_path)

    #验证集的处理
    for image_path in glob.glob(a.val_images_path_jpg): #每一张图片都对应一个xml文件这里写xml对应的图片的路径
        image_name = image_path.split('\\')[-1]
        convert_annotation(image_name,a.val_labels_path,a.val_labels_xml_path)
