from CycleGAN_VC2 import Generator, Discriminator
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

netG_A2B = Generator(num_features=94)
netG_A2B2 = Generator(num_features=94)
netG_A2B3 = Generator(num_features=94)
#netG_B2A = Generator(num_features=94)
netD_A = Discriminator(64, 94)
#netD_B = Discriminator(64, 94)

if torch.cuda.is_available():
    netG_A2B.cuda()
    netG_A2B2.cuda()
    netG_A2B3.cuda()
    #netG_B2A.cuda()
    netD_A.cuda()
    #netD_B.cuda()


netG_A2B.load_state_dict(torch.load('output/netG_IU2LJB.pth'))
netG_A2B2.load_state_dict(torch.load('output/netG_IU2LJB2.pth'))
netG_A2B3.load_state_dict(torch.load('output/netG_IU2LJB3.pth'))
#netG_B2A.load_state_dict(torch.load('output/netG_LJB2IU.pth'))
#netG_A2B.load_state_dict(torch.load('output/netG_IU2LJB.pth'))
#netG_A2B.load_state_dict(torch.load('output/netG_IU2LJB.pth'))
#netG_A2B.load_state_dict(torch.load('output/netG_IU2LJB.pth'))
netD_A.load_state_dict(torch.load('output/netD_IU.pth'))
#netD_B.load_state_dict(torch.load('output/netD_B.pth'))

file_dir = "data/IU/wav"
filename = "GoodDay.wav"

y, sr = librosa.load(file_dir + "/" + filename, sr=16000)
#original = []
result = []
result2 = []
result3 = []
ensemble = []
for i in range(1, 11):
    yt = y[sr*i*3:sr*3*(i+1)]
    mfccs = librosa.feature.mfcc(yt, n_mfcc=64)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('input')
    plt.tight_layout()
    plt.show()
    mfcc = mfccs
    mfccs = mfccs.reshape((1, 1, mfccs.shape[0], mfccs.shape[1]))
    mfccs = torch.from_numpy(mfccs).float().cuda()
    output = netG_A2B(mfccs)
    output2 = netG_A2B2(mfccs)
    output3 = netG_A2B3(mfccs)
    print(netD_A(mfccs))
    print(netD_A(output))
    output = output.cpu().detach().numpy()
    output2 = output2.cpu().detach().numpy()
    output3 = output3.cpu().detach().numpy()
    output = output.reshape((output.shape[2], output.shape[3]))
    output2 = output2.reshape((output2.shape[2], output2.shape[3]))
    output3 = output3.reshape((output3.shape[2], output3.shape[3]))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(output, x_axis='time')
    plt.colorbar()
    plt.title('output')
    plt.tight_layout()
    plt.show()
    audio = librosa.feature.inverse.mfcc_to_audio(output)
    audio2 = librosa.feature.inverse.mfcc_to_audio(output2)
    audio3 = librosa.feature.inverse.mfcc_to_audio(output3)
    #originals = librosa.feature.inverse.mfcc_to_audio(mfcc)
    for t in audio:
        result.append(t)
    for t in audio2:
        result2.append(t)
    for t in audio3:
        result3.append(t)
    for s in range(len(audio)):
        ensemble.append((audio[s] + audio2[s] + audio3[s]) / 3)
    #for k in originals:
        #original.append(k)
print(result)
result = np.asarray(result)
result2 = np.asarray(result2)
result3 = np.asarray(result3)
ensemble = np.asarray(ensemble, dtype=np.float32)
print(type(ensemble[0]))
#original = np.asarray(original)
print(result[0])
librosa.output.write_wav("test.wav", result, sr=16000)
librosa.output.write_wav("test2.wav", result2, sr=16000)
librosa.output.write_wav("test3.wav", result3, sr=16000)
librosa.output.write_wav("ensemble.wav", ensemble, sr=16000)
#librosa.output.write_wav("original.wav", original, sr=16000)
