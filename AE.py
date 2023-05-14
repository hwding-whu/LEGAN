
import torch
from tqdm import trange
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
#参数设置
batch_size = 64
num_epochs = 200
learning_rate = 1*1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = './logs/pathmnist/sample/'
if os.path.exists(img_path): True
else: os.makedirs(img_path)

model_path = './logs/pathmnist/model/'
if os.path.exists(model_path): True
else: os.makedirs(model_path)

data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
])
train_ds = ImageFolder("./data/pathmnist/Training", transform=data_transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x_e = self.encoder(x)
        x_d = self.decoder(x_e)
        return x_e, x_d


model = autoencoder().to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 28, 28)
    return x

epoch = 0
with trange(1, num_epochs + 1, desc='Training', ncols=0) as pbar:
    for step in pbar:
        total_loss = 0
        for data in train_dl:
            img, _ = data
            img = Variable(img).to(device)
            # ===================forward=====================
            x_e, x_d = model(img)
            loss = criterion(x_d, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, total_loss))
        if epoch % 10 == 0 or epoch == num_epochs:
            pic = to_img(x_d.cpu().data)
            save_image(pic, img_path+'image_{}.png'.format(epoch))
        epoch += 1


#torch.save(model.state_dict(), './save_model.pth')
torch.save(model.encoder, model_path+'encoder_model.pth')
torch.save(model.decoder, model_path+'decoder_model.pth')