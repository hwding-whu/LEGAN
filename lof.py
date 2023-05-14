import shutil
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
from torchvision.datasets import ImageFolder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor as LOF
import numpy as np
from os.path import normpath, basename
import os
import turtle as t

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = 'logs/pathmnist/model'
encoder = torch.load(path+'/encoder_model.pth')

data_transform = transforms.Compose([
        # transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
train_ds = ImageFolder("./data/pathmnist/8", transform=data_transform)
# for i in range(len(train_ds)):print(basename(normpath(train_ds.imgs[i][0])))
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
save_path = "./data/pathmnist/outline/8/"
if os.path.exists(save_path): True
else: os.makedirs(save_path)
# 训练集
imgs_tr, _ = next(iter(train_dl))
img = Variable(imgs_tr).to(device)
x_e = encoder(img)
train = x_e.cpu().view(-1, 4 * 4 * 32).detach().numpy()
pca = PCA(n_components=2, random_state=42)
pca = pca.fit(train)
train = pca.transform(train)
# LOF
clf = LOF(n_neighbors=20)
cl = clf.fit_predict(train)
index = [i for i, x in enumerate(cl.tolist()) if x == -1]
print(len(index))
# 画图
plt.scatter(train[:, 0], train[:, 1], c='b')
plt.title("Visual clustering")
plt.show()
# 画图
cm = matplotlib.colors.ListedColormap(['r', 'b'])
plt.scatter(train[:, 0], train[:, 1], c=cl, cmap=cm)
plt.title("Visual clustering")
plt.show()
# 生成离群数据
n = len(cl)//len(index)-1
print(n)
t = 0
for i in range(0, 1):
    for j in range(len(index)):
        shutil.move(train_ds.imgs[index[j]][0], save_path + '%d.png'%(t))
        t = t+1


