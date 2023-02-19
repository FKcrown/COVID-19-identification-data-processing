import numpy as np
import chirplet as ch
import matplotlib.pyplot as plt
import librosa.util
import librosa
import os
import scipy.signal as sg
import soundfile as sf
import csv
import warnings
from warnings import simplefilter


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def compute_spectrogram(signal, sample_rate):
    """
        compute spectrogram from signal
        :param signal:
        :return: matrice representing the spectrum, frequencies corresponding and times
    """
    [spectrum, freqs, times] = plt.mlab.specgram(signal, NFFT=1024, Fs=sample_rate,
                                                 noverlap=512, window=np.hamming(1024))
    spectrum = 10. * np.log10(spectrum)

    return [spectrum, freqs, times]


def compute_TFDF(signal):
    freq = np.size(signal, 0)  # 计算 X 的行数
    time = np.size(signal, 1)  # 计算 X 的列数

    TFDF_array = [[]]
    for m in range(freq - 1):
        TFDF_array.append([])
        for t in range(time - 1):
            dE_FT = signal[m, t + 1] + signal[m + 1, t] - signal[m, t] - signal[m + 1, t + 1]
            TFDF_array[m].append(dE_FT)
    TFDF_array.remove([])
    TFDF = np.array(TFDF_array)

    return TFDF


def plotchirplet(chirps, audiopath):
    # print("--- filename---" ,audiopath)
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)
    # print('绘制chirp' )
    # cmap = cmx.gray   #jet,parula,gray,rainbow
    figure, axarr = plt.subplots(1, sharex=False)
    # glength=len(data)/sr*80
    figure.set_size_inches(4, 4)
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    tabfinal = list(reversed(chirps))


    [spectrum, freqs, times] = compute_spectrogram(data, sr)

    index_frequency = np.argmax(freqs)
    mxf = freqs[index_frequency]
    # print("最大频率",mxf)

    axarr.matshow(tabfinal, cmap='jet',
                  origin='lower',
                  extent=(0, times[-1], freqs[0], mxf),
                  aspect='auto')

    # plt.axis('off')

    # ch_folder = audiopath[: audiopath.rfind("\\")] + str(r'\chirplet')
    #
    # create_dir_not_exist(ch_folder)
    # figure.savefig(ch_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'), dpi=56)
    plt.show()

    # plt.close('all')


file = r'F:\Database\Audios\整合\positive\vad\new\00af1e34-c02b-4f68-a5e5-3100db5e31f7-16.0K-VAD-0.wav'

data, sr = librosa.load(file, sr=None)
# data=data/abs(data).max()#对语音进行归一化
ch1 = ch.FCT(sample_rate=sr)
chirps = ch1.compute(data)
# plotchirplet(data, chirps)


'''
'''
figure, axarr = plt.subplots(1, sharex=False)
figure.set_size_inches(4, 4)
# 改变图像边沿大小，参数分别为左下右上，子图间距
figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
temp = list(reversed(chirps))
tabfinal = np.array(temp).reshape(len(temp), -1)
TFDF = compute_TFDF(tabfinal)


[spectrum, freqs, times] = plt.mlab.specgram(data, NFFT=1024, Fs=sr,
                                                 noverlap=512, window=np.hamming(1024))

index_frequency = np.argmax(freqs)
mxf = freqs[index_frequency]


axarr.matshow(TFDF, cmap='jet',
              origin='lower',
              extent=(0, times[-1], freqs[0], mxf),
              # extent=(logmelspec[])
              aspect='auto')

# librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.show()
