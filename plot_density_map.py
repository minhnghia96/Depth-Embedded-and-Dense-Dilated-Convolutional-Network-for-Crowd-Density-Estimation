# import matplotlib.pyplot as plt
# import scipy.io
# from scipy.stats.kde import gaussian_kde
# import numpy as np

# data = scipy.io.loadmat('/mnt/d/Master/Thesis/Code/Thesis_Crowd_Counting/test_data/IMG_5.mat', squeeze_me=True)
# data = data['image_info']['location'].item()

# x = []
# y = []
# for i in data:
#     x.append(i[0])
#     y.append(i[1])
# x = np.array(x)
# y = np.array(y)

# y = y[np.logical_not(np.isnan(y))]
# x = x[np.logical_not(np.isnan(x))]
# k = gaussian_kde(np.vstack([x, y]))
# xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# # fig = plt.figure(figsize=(7,8))
# # ax1 = fig.add_subplot(211)
# # ax2 = fig.add_subplot(212)

# # # alpha=0.5 will make the plots semitransparent
# # ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
# # ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

# # ax1.set_xlim(x.min(), x.max())
# # ax1.set_ylim(y.min(), y.max())
# # ax2.set_xlim(x.min(), x.max())
# # ax2.set_ylim(y.min(), y.max())

# # # you can also overlay your soccer field
# # im = plt.imread('/mnt/d/Master/Thesis/Code/Thesis_Crowd_Counting/test_data/IMG_5.jpg')
# # ax1.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
# # ax2.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

# fig = plt.figure(figsize=(9,10))
# ax1 = fig.add_subplot(211)


# ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)

# ax1.plot(x, y,"o")
# ax1.set_xlim(x.min(), x.max())
# ax1.set_ylim(y.min(), y.max())

# #overlay soccer field
# # im = plt.imread('/mnt/d/Master/Thesis/Code/Thesis_Crowd_Counting/test_data/IMG_5.jpg')
# # ax1.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

# # plt.plot(x, linestyle='', marker='x')
# fig.savefig('test.jpg')2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM

img_path = '/media/tma/DATA/NghiaNguyen/Thesis_Crowd_Counting/dataset/ShanghaiTech/part_B/train_data/images/IMG_1.jpg'
# gt_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'),'r')
gt_file = h5py.File(img_path.replace('.jpg','.h5').replace("images", "ground-truth"),'r')
groundtruth = np.asarray(gt_file['density'])
# groundtruth = groundtruth.squeeze().detach().numpy()
print(groundtruth)
height, width = groundtruth.shape
fig, ax = plt.subplots()
ax.imshow(groundtruth, cmap=CM.jet)
fig.set_size_inches(width/100.0, height/100.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.axis('off')
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0)
plt.text(x=20,y=20,s='GT count: {}'.format(round(groundtruth.sum())), fontsize=20,color='white')
plt.savefig(img_path.replace('.jpg', '_ground_truth.jpg'), dpi=300)
plt.show()