import math
import numpy as np
import torch
import cv2

def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    print("r的值是：",r)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels



if __name__ == "__main__":
    # x = 9.01
    # x = math.ceil(x)
    # print(x)
    # img_1 = torch.randn(2,3,3)
    # img_2 = torch.randn(2,3,3)
    # label_1 = torch.tensor([[0,0,1]])
    # label_2 = torch.tensor([[0,1,0]])
    # print(label_1.shape)
    # labels = np.concatenate((label_1, label_2), 0)
    # print(labels.shape)
    img = np.array([[[2,3,3],
                    [1,3,6],
                    [4,3,9],
                    [7,2,3]],
                    [[2,3,3],
                    [1,3,6],
                    [4,3,9],
                    [7,2,3]]])
    print(img.shape)
    im = cv2.flip(img,1)
    print(im.shape)
    pass
