import h5py
import numpy as np
import pydub
import os
from pydub import AudioSegment
import vggish_input
import librosa

def load_and_save(file_dir="./", save_name="default.h5", key_name="default", crop_time=4):
    hf = h5py.File(file_dir + save_name, "w")
    song_array = []  # song은 time, wave
    for file in os.listdir(file_dir):
        if file.endswith(".mp3"):
            song = pydub.AudioSegment.from_mp3(file_dir + "/" + file)

            if song.channels == 2:
                song = np.array(song.get_array_of_samples())
                song = song.reshape((-1, 2))

                for i in range(2):
                    sec3 = []
                    for sec in range(1, len(song)):
                        sec3.append(song[sec][i])

                        if sec % (44100 * crop_time) == 0:
                            song_array.append(sec3)
                            sec3 = []


            else:
                song = np.array(song.get_array_of_samples())
                song = song.reshape((-1, 1))

                sec3 = []
                for sec in range(1, len(song)):
                    sec3.append(song[sec])

                    if sec % (44100 * crop_time) == 0:
                        song_array.append(sec3)
                        sec3 = []


        if file.endswith(".wav"):
            song = pydub.AudioSegment.from_mp3(file_dir + "/" + file)
            song = np.array(song.get_array_of_samples())
            song = song.reshape((-1, 1))

            sec3 = []
            for sec in range(1, len(song)):
                sec3.append(song[sec])

                if sec % (44100 * crop_time) == 0:
                    song_array.append(sec3)
                    sec3 = []

    song_array = np.asarray(song_array).reshape((-1, 44100 * crop_time, 1)).tolist()
    print(np.asarray(song_array).shape)

    print(np.asarray(song_array[0]).shape)
    print(np.asarray(song_array[1]).shape)
    hf.create_dataset(key_name, data=song_array)
    hf.close()


def mp3_to_wav(file_dir="", save_file_dir=""):
    filenames = os.listdir(file_dir)
    for filename in filenames:
        fname, ext = os.path.splitext(filename) # 확장자 제거
        dst = fname + '.wav'

        sound = AudioSegment.from_mp3(file_dir + "/" + filename)
        sound.export(save_file_dir + "/" + dst, format="wav")


def wav_to_hdf5(file_dir="", save_file=""):
    filenames = os.listdir(file_dir)
    hf = h5py.File(save_file, "w")
    song_array = []
    for filename in filenames:
        y, sr = librosa.load(file_dir + "/" + filename, sr=16000)
        sec = int(sr * 3)
        for i in range(int(y.shape[0]/sec)):
            batch = y[i*sec:(i+1)*sec]
            mfccs = librosa.feature.mfcc(np.asarray(batch), n_mfcc=64)
            song_array.append(mfccs)
    song_array = np.asarray(song_array)
    print(song_array.shape)
    hf.create_dataset("song", data=song_array)
    hf.close()


wav_to_hdf5("./data/IU/wav/", "IU.h5")
#mp3_to_wav("./data/LJB/mp3", "./data/LJB/wav")