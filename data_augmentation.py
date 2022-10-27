import imgaug.augmenters as iaa
import cv2
import glob, os
import shutil
from tqdm import tqdm
import scipy.io as scio
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
import itertools

def crop():
    image_path = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_A_augmented/train_data/images'
    label_path = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_A_augmented/train_data/ground-truth'
    image_save_path = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_A_augmented_cropped/train_data/images'
    label_save_path = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_A_augmented_cropped/train_data/ground-truth'
    crop_size = 512

    image_files = [filename for filename in os.listdir(image_path) \
                       if os.path.isfile(os.path.join(image_path,filename)) and '.jpg' in filename]

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    for image_file in tqdm(image_files):
        label_file=image_file.replace('.jpg','.mat').replace('IMG_','GT_IMG_')
        # label_file=image_file.replace('.jpg','_ann.mat')
        img = imread(os.path.join(image_path, image_file), 0)
        img = img.astype(np.float32, copy=False)
        hh = img.shape[0]
        ww = img.shape[1]
        # annPoints = scio.loadmat((os.path.join(label_path, label_file)))['annPoints']
        annPoints = scio.loadmat((os.path.join(label_path, label_file)))["image_info"][0,0][0,0][0]
        if hh < crop_size or ww < crop_size:
            imsave(os.path.join(image_save_path, image_file.split('.')[0] + "_%d.jpg" % i), img)
            scio.savemat(os.path.join(label_save_path, image_file.split('.')[0] + "_%d_ann.mat" % i), {"annPoints": annPoints})
            continue
        h_pad = crop_size - hh % crop_size
        w_pad = crop_size - ww % crop_size
        h_slice = [crop_size * i for i in range(hh // crop_size + 1)]
        w_slice = [crop_size * i for i in range(ww // crop_size + 1)]
        h_border = [h_pad // (hh // crop_size)] * (hh // crop_size - 1) + [h_pad - h_pad // (hh // crop_size) * (hh // crop_size - 1)]
        w_border = [w_pad // (ww // crop_size)] * (ww // crop_size - 1) + [w_pad - w_pad // (ww // crop_size) * (ww // crop_size - 1)]
        h_border = [0] + np.cumsum(h_border).tolist()
        w_border = [0] + np.cumsum(w_border).tolist()
        # print h_pad, h_border, w_pad, w_border
        h_indexs = np.array(h_slice) - np.array(h_border)
        w_indexs = np.array(w_slice) - np.array(w_border)
        # print h_index, w_index
        sum_count = 0
        for i, (h_index, w_index) in enumerate(itertools.product(h_indexs, w_indexs)):
            patch = img[h_index:h_index + crop_size, w_index:w_index + crop_size,...]
            count = annPoints[((annPoints[:,0] >= w_index) & (annPoints[:,0] < w_index + crop_size) & \
                               (annPoints[:,1] >= h_index) & (annPoints[:,1] < h_index + crop_size) )]
            # print (w_index, w_index + crop_size, h_index, h_index + crop_size, count, '==' * 10) 
            count[:, 0] = count[:, 0] - w_index
            count[:, 1] = count[:, 1] - h_index
            # print (patch.shape, np.size(count))
            sum_count += np.size(count)
            imsave(os.path.join(image_save_path, image_file.split('.')[0] + "_%d.jpg" % i), patch)
            scio.savemat(os.path.join(label_save_path, image_file.split('.')[0] + "_%d_ann.mat" % i), {"annPoints": count})
            
        # print (img.shape, annPoints.shape, annPoints[:,0].max(), annPoints[:,1].min())
        # print ("*" * 30, sum_count)
def augment():
    add_noise = iaa.AdditiveGaussianNoise(scale=(0, 0.3*255))
    blur = iaa.GaussianBlur(sigma=(0.0, 3.0))
    flip = iaa.Fliplr(1.0)

    augment_types = ['add_noise', 'blur']

    img_dir = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_A/train_data/images'
    save_dir = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_A_augmented/train_data/images'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir.replace('images', 'ground-truth'))

    for img_path in tqdm(glob.glob(os.path.join(img_dir, '*.jpg'))):
        img_name = os.path.basename(img_path).split(".")[0]
        mat_path_src = img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_')
        img = cv2.imread(img_path)
        for augment_type in augment_types:
            if augment_type == "add_noise":
                img_added_noise = add_noise(image=img)
                img_save_path = os.path.join(save_dir, img_name+'_'+augment_type+'.jpg')
                cv2.imwrite(img_save_path, img_added_noise)

                mat_name = os.path.basename(mat_path_src).split(".")[0]
                mat_path_dst = os.path.join(save_dir, mat_name+'_'+augment_type+'.mat').replace('images', 'ground-truth')
                shutil.copyfile(mat_path_src, mat_path_dst)
            elif augment_type == "blur":
                img_blured = blur(image=img)
                img_save_path = os.path.join(save_dir, img_name+'_'+augment_type+'.jpg')
                cv2.imwrite(img_save_path, img_blured)

                mat_name = os.path.basename(mat_path_src).split(".")[0]
                mat_path_dst = os.path.join(save_dir, mat_name+'_'+augment_type+'.mat').replace('images', 'ground-truth')
                shutil.copyfile(mat_path_src, mat_path_dst)
            elif augment_type == "flip":
                img_flipped = flip(image=img)
                img_save_path = os.path.join(save_dir, img_name+'_'+augment_type+'.jpg')
                cv2.imwrite(img_save_path, img_flipped)

                mat_name = os.path.basename(mat_path_src).split(".")[0]
                mat_path_dst = os.path.join(save_dir, mat_name+'_'+augment_type+'.mat').replace('images', 'ground-truth')
                shutil.copyfile(mat_path_src, mat_path_dst)
            else:
                continue

if __name__ == '__main__':
    # augment()
    crop()
