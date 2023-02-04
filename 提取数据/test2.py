import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.util
import librosa
import os
import scipy.signal as sg
import soundfile as sf

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

def plotmelspec(audiopath):
    print(audiopath)
    # print("--- filename---" ,audiopath)
    # print('绘制spec' )
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)

    # cmap = cmx.gray   #jet,parula,gray,rainbow
    figure, axarr = plt.subplots(1, sharex=False)
    # glength=len(data)/sr*80
    figure.set_size_inches(4, 4)  # 输出图大小和时间长度成正比50ms一幅图

    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)

    max_frequency = freqs[index_frequency]

    melspec = librosa.feature.mfcc(data, sr, n_mfcc=32, n_fft=1024, hop_length=512, power=2.0)  # 计算mel倒谱

    # print('mel',np.shape(melspec))

    axarr.matshow(melspec[1:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], freqs[0], max_frequency),
                  aspect='auto')

    plt.axis('off')

    plt.close('all')


audio_path = r'F:\Database\Audios\整合\negative\vad\new\aADACozj_cough-16.0K-VAD-0.wav'
data, sr = librosa.load(audio_path)

data_1 = librosa.resample(data, sr, 16000)



figure, axarr = plt.subplots(1, sharex=False)
figure.set_size_inches(4, 4)
# 改变图像边沿大小，参数分别为左下右上，子图间距
figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

[spectrum, freqs, times] = plt.mlab.specgram(data_1, NFFT=1024, Fs=sr,
                                                 noverlap=512, window=np.hamming(1024))
melspec = librosa.feature.mfcc(data, sr, n_mfcc=32, n_fft=1024, hop_length=512)
logmelspec = librosa.power_to_db(melspec)

TFDF = compute_TFDF(logmelspec)

index_frequency = np.argmax(freqs)
max_frequency = freqs[index_frequency]

axarr.matshow(TFDF[0:index_frequency, :], cmap='jet',
              origin='lower',
              extent=(times[0], times[-1], freqs[0], max_frequency),
              # extent=(logmelspec[])
              aspect='auto')

# librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.show()












