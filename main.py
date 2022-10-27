import os
import glob
import logging
import warnings
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import RawDataset
from KITTIDepth.dataset import DataGenerator
from modeling.new_network import DenseScaleNet as DSNet
warnings.filterwarnings("ignore")

def cal_lc_loss(output, target, sizes=(1,2,4)):
    criterion_L1 = nn.L1Loss()
    Lc_loss = None
    for s in sizes:
        pool = nn.AdaptiveAvgPool2d(s)
        est = pool(output)
        gt = pool(target)
        if Lc_loss:
            Lc_loss += criterion_L1(est, gt)
        else:
            Lc_loss = criterion_L1(est, gt)
    return Lc_loss


def getLogger(filename):
    logger = logging.getLogger('train_logger')

    while logger.handlers:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s], ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
    

def val(model, test_loader, device):
    model.eval()
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for (img, _, count) in test_loader:
            img = img.to(device)
            output1 = model(x1=img, phase='val')
            est_count = output1.sum().item()
            mae += abs(est_count - count)
            mse += (est_count - count)**2
    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse**0.5
    return float(mae), float(mse)

def get_loader(train_path, test_path, ratio):
    train_img_paths = []
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        train_img_paths.append(img_path)
    
    test_img_paths = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_img_paths.append(img_path)
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_loader = torch.utils.data.DataLoader(
        RawDataset(
            train_img_paths[:3381], 
            transform, 
            aug=True, 
            ratio=ratio), 
        shuffle=True, 
        batch_size=1)
    test_loader = torch.utils.data.DataLoader(
        RawDataset(
            test_img_paths, 
            transform, 
            ratio=1, 
            aug=False), 
        shuffle=False, 
        batch_size=1)
    
    return train_loader, test_loader


def main():
    # torch.cuda.set_per_process_memory_fraction(0.1, 0)
    # Init
    dens_dataset = 'sha'
    filename = 'MyNetwork_0415_SHT_A'
    pretrained_weight_path = "/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/trained_models/DenseScaleNet_sha_1e-5_1219.pth"
    if not os.path.exists("trained_models/"+filename):
        os.makedirs("trained_models/"+filename)
    best_weigth_save_path = os.path.join("trained_models/"+filename, filename+'.pth')
    checkpoint_save_path = os.path.join("trained_models/"+filename, filename+'_checkpoint.pth')
    log_save_path = os.path.join("trained_models/"+filename, filename+'.txt')
    epochs = 300
    criterion = nn.MSELoss()
    logger = getLogger(log_save_path)

    # Select device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))
    
    # Init model
    net = DSNet(pretrained_model=pretrained_weight_path)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-6, weight_decay=5e-4)


    net.to(device)

    # Data Dir
    if dens_dataset == "sha":
        train_path = 'dataset/ShanghaiTech/part_A_full_augment/train_data/images/'
        test_path = 'dataset/ShanghaiTech/part_A/test_data/images/'
    elif dens_dataset == "shb":
        train_path = 'dataset/ShanghaiTech/part_B_full/train_data/images/'
        test_path = 'dataset/ShanghaiTech/part_B/test_data/images/'
    elif dens_dataset == "ucf":
        train_path = 'dataset/UCF-Train-Val-Test/train/images/'
        test_path = 'dataset/UCF-Train-Val-Test/val/images/'
    kittiDir = "dataset/kitti_dataset"

    # Load Data
    dens_train_loader, dens_test_loader = get_loader(train_path, test_path, 8)

    train_kitti_generator  = DataGenerator(kittiDir, "train", True)
    train_kitti_loader = train_kitti_generator.create_data(batch_size=1)

    print('dens: {}'.format(len(dens_train_loader)))
    print('depth: {}'.format(len(train_kitti_loader)))
  
    best_mae, _  = val(net, dens_test_loader, device)
    for epoch in range(epochs):
        train_loss = 0.0
        net.train()
        for (img, target, _), kitti_data, (pMap_img, pMap) in tqdm(
            zip(dens_train_loader, train_kitti_loader), total=len(dens_train_loader)):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            
            kitti_img = kitti_data["img"].to(device)
            kitti_depth = kitti_data["depth"].to(device)

            output1, output2 = net(img, kitti_img)

            Le_Loss1 = criterion(output1, target)
            Lc_Loss1 = cal_lc_loss(output1, target)
            loss1 = Le_Loss1 + 1000 * Lc_Loss1
            Le_Loss2 = criterion(output2, kitti_depth)
            Lc_Loss2 = cal_lc_loss(output2, kitti_depth)
            loss2 = Le_Loss2 + 1000 * Lc_Loss2
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        mae, mse = val(net, dens_test_loader, device)
        
        logger.info('Epoch {}/{} Loss:{:.3f}, MAE:{:.2f}, MSE:{:.2f}, Best MAE:{:.2f}'.format(
            epoch+1, epochs, train_loss/len(dens_train_loader), mae, mse, best_mae))
        
        # Save Checkpoint
        torch.save(net.state_dict(), checkpoint_save_path)
        
        if mae < best_mae:
            best_mae = mae
            torch.save(net.state_dict(), best_weigth_save_path)


if __name__ == '__main__':
    main()