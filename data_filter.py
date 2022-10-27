from shutil import copyfile
import os
from distutils.dir_util import copy_tree
with open('/mnt/d/Master/Thesis/Data/kitti_dataset_note.txt', 'r+') as f:
    lines=f.read().splitlines()
    f.close()

datapath="/mnt/e/Data/Crowd_Counting/kitti/data_depth_annotated/val"
filter_datapath="/mnt/e/Data/Crowd_Counting/kitti/data_depth_annotated_filter/val"
if not os.path.exists(filter_datapath):
    os.makedirs(filter_datapath)

for line in lines:
    # date = line.split('_drive')[0]
    # path = os.path.join(datapath, line+"_sync")
    # save_path = os.path.join(filter_datapath, line+"_sync")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    for foldername in os.listdir(datapath):
        if foldername == line+"_sync":
            fromDirectory = os.path.join(datapath, foldername)
            toDirectory = os.path.join(filter_datapath, foldername)
            if not os.path.exists(toDirectory):
                os.makedirs(toDirectory)
            print(foldername)
            copy_tree(fromDirectory, toDirectory)
    #     if foldername == "image_02" or foldername == "image_03":
    #         if not os.path.exists(os.path.join(save_path, foldername, 'data')):
    #             os.makedirs(os.path.join(save_path, foldername, 'data'))
    #         for filename in os.listdir(os.path.join(path, foldername, 'data')):
    #             src=os.path.join(path, foldername, 'data', filename)
    #             dst=os.path.join(save_path, foldername, 'data', filename)
    #             copyfile(src, dst)
    #             print(dst)
    # print(line_splitted)

