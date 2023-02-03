import numpy as np
import chirplet as ch
import matplotlib.pyplot as plt
import librosa.util
import librosa
import os
import scipy.signal as sg
import soundfile as sf
import csv


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


def plotsft(audiopath):
    # print("--- filename---" ,audiopath)
    # print('绘制sft' )
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)

    figure, axarr = plt.subplots(1, sharex=False)
    figure.set_size_inches(4, 4)
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)
    mxf = freqs[index_frequency]
    # print("最大频率",mxf)

    axarr.matshow(spectrum[0:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], freqs[0], mxf),
                  aspect='auto')
    plt.axis('off')
    # axarr.axes.xaxis.set_ticks_position('bottom')
    # axarr.set_ylabel("Frequency in Hz")
    # axarr.xaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)
    # axarr.yaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)

    # axarr.set_title('spectrogram')

    # figure.tight_layout()

    figure.savefig(audiopath[:-4] + str('spe') + str('.jpg'), dpi=100)
    # plt.show()

    plt.close('all')
