import torch
import torch.nn as nn
from CycleGAN_VC2 import Generator, Discriminator
import vggish
import vggish_input
from torch.utils.data import Dataset, DataLoader
import h5py
from torch.autograd import Variable
import time


singerA_dataset = "./IU.h5"
singerB_dataset = "./LJB.h5"
batchSize = 32
load_model = True


if torch.cuda.is_available():
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


class SongDataset(Dataset):
    def __init__(self, filename):
        self.data = h5py.File(filename, 'r')
        self.song = self.data.get("song").value
        self.x = self.song.reshape((-1, 1, 64, 94))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


train_A = SongDataset(singerA_dataset)
train_B = SongDataset(singerB_dataset)

train_loader_A = DataLoader(dataset=train_A, batch_size=batchSize, shuffle=True)
train_loader_B = DataLoader(dataset=train_B, batch_size=batchSize, shuffle=True)

netG_A2B = Generator(num_features=94)
netG_B2A = Generator(num_features=94)
netD_A = Discriminator(64, 94)
netD_B = Discriminator(64, 94)

if torch.cuda.is_available():
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

if load_model:
    netG_A2B.load_state_dict(torch.load('output/netG_IU2LJB.pth'))
    netG_B2A.load_state_dict(torch.load('output/netG_LJB2IU.pth'))
    netD_A.load_state_dict(torch.load('output/netD_IU.pth'))
    netD_B.load_state_dict(torch.load('output/netD_LJB.pth'))

criterion_GAN = torch.nn.BCELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(list(netG_A2B.parameters()) + list(netG_B2A.parameters()), lr=0.001)
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.001)
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.001)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(batchSize, 1, 64, 94)
input_B = Tensor(batchSize, 1, 64, 94)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)


###################################

###### Training ######
for epoch in range(1000):
    start = time.time()
    for i, (real_A, real_B) in enumerate(zip(train_loader_A, train_loader_B)):

        if real_A.shape[0] != batchSize:
            break
        if real_B.shape[0] != batchSize:
            break

        real_A = real_A.cuda().float()
        real_B = real_B.cuda().float()

        ############ Generator ############
        optimizer_G.zero_grad()

        netG_A2B.train()
        netG_B2A.train()
        netD_A.eval()
        netD_B.eval()

        # gan loss
        G_A2B = netG_A2B(real_A)
        G_B2A = netG_B2A(real_B)
        pred_fake_B = netD_B(G_A2B)
        pred_fake_A = netD_A(G_B2A)
        gan_loss_A = criterion_GAN(pred_fake_A, target_real)
        gan_loss_B = criterion_GAN(pred_fake_B, target_real)

        gan_loss = (gan_loss_A + gan_loss_B) * 10

        # identity loss A를 넣으면 A가, B를 넣으면 B가 나오도록 identity를 설정
        G_A2A = netG_B2A(real_A)
        G_B2B = netG_A2B(real_B)
        identity_loss_A = criterion_identity(G_A2A, real_A)
        identity_loss_B = criterion_identity(G_B2B, real_B)

        identity_loss = (identity_loss_A + identity_loss_B) * 5

        # cycle loss
        G_A2B2A = netG_B2A(G_A2B)
        G_B2A2B = netG_A2B(G_B2A)

        cycle_loss_A = criterion_cycle(G_A2B2A, real_A)
        cycle_loss_B = criterion_cycle(G_B2A2B, real_B)

        cycle_loss = (cycle_loss_A + cycle_loss_B) * 10

        # cycle loss with discriminator
        pred_fake_B2A2B = netD_B(G_B2A2B)
        pred_fake_A2B2A = netD_A(G_A2B2A)

        cycle_gan_loss_A = criterion_GAN(pred_fake_A, target_real)
        cycle_gan_loss_B = criterion_GAN(pred_fake_B, target_real)

        cycle_gan_loss = (cycle_gan_loss_A + cycle_gan_loss_B) * 10

        # Total loss
        loss_G = identity_loss + gan_loss + cycle_loss + cycle_gan_loss
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.train()
        netD_B.train()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        pred_fake = netD_A(netG_B2A(real_B))
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_B(netG_A2B(real_A))
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
    print("time :", time.time() - start)
    print("epoch ", epoch+1, " finished")

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_IU2LJB.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_LJB2IU.pth')
    torch.save(netD_A.state_dict(), 'output/netD_IU.pth')
    torch.save(netD_B.state_dict(), 'output/netD_LJB.pth')
###################################