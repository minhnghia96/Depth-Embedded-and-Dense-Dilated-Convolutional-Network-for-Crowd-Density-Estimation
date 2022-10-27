import cv2, glob
import numpy as np
import time
import torch
from torch.autograd import Variable
from modeling.new_network import DenseScaleNet as DSNet
import os
from matplotlib import cm as CM
import matplotlib.pyplot as plt
import argparse
import scipy.io

def preprocess_image(cv2im):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

    
def img_test(pretrained_model, img_path):
    img = cv2.imread(img_path)
    # print("Image Shape: {}".format(img.shape))
    # img_resized=cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    img = preprocess_image(img)
    if torch.cuda.is_available():
        img = img.cuda()
        # pretrained_model = pretrained_model.cuda()
    outputs = pretrained_model(img)
    if torch.cuda.is_available():
        dmp = outputs[0].squeeze().detach().cpu().numpy()
        # amp = outputs[-1].squeeze().detach().cpu().numpy()
    else:
        dmp = outputs[0].squeeze().detach().numpy()
        # amp = outputs[-1].squeeze().detach().numpy()
    dmp = dmp
    print('Estimated people count: ', round(dmp.sum()))
    return dmp


def main(args):
    # img_dir ="/media/tma/DATA/NghiaNguyen/github/Bayesian-Crowd-Counting/UCF-QNRF_ECCV18/Test/*.jpg"
    model = DSNet(args.model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    # img_count = 0
    # total_process_time=0
    # for img_path in glob.glob(img_dir):
        # print(img_path)
        # img_count+=1
    start_time=time.time()
    dmp = img_test(model, args.test_img_path)
    end_time=time.time()
    print("Processing Time: {}".format(end_time-start_time))
    # total_process_time+=(end_time-start_time)
    # print("Average Process Time: {}".format(total_process_time/img_count))
    # print(dmp)
    height, width = dmp.shape
    fig, ax = plt.subplots()
    ax.imshow(dmp, cmap=CM.jet)
    fig.set_size_inches(width/100.0, height/100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.text(x=5,y=5,s='Est count: {}'.format(round(dmp.sum())), fontsize=5,color='white')
    plt.savefig(args.test_img_path.replace('.jpg', '_density_map_SHB.jpg'), dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense Scale Net Image Test')
    parser.add_argument('--gpu', default='0', help='assign device')
    parser.add_argument('--model_path', metavar='model path', 
    default="trained_models/MyNetwork_0307_SHT_A_cropped/MyNetwork_0307_SHT_A_cropped.pth", type=str)
    parser.add_argument('--test_img_path', metavar='test image path', type=str)
    
    args = parser.parse_args()
    ### Load grounth truth
    print(os.path.basename(args.test_img_path))
    img_name = os.path.basename(args.test_img_path)
    mat_name = ("GT_" + img_name).replace('.jpg', '.mat')
    mat_path = args.test_img_path.replace(img_name, mat_name).replace("images", "ground-truth")
    print(mat_path)
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)
    print('Number of people in ground truth: ', mat['image_info']['number'])
    main(args)
