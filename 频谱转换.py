#! /usr/bin/env python3
import numpy as np
from 整合 import chirplet as ch
import matplotlib.pyplot as plt
import os
import wave
import struct


def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window


def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧,且去掉过短的信号
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    winfunc窗函数winfunc = signal.hamming(nw)
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
        return None, nf
    else:  # 否则，计算帧的总长度
        nf = int(np.floor((1.0 * signal_length - nw + inc) / inc))
        whole_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
        pro_signal = signal[0: whole_length]  # 截去后的信号记为pro_signal
        indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                               (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = pro_signal[indices]  # 得到帧信号
        win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
        print("enframe finished")
        return frames * win, nf  # 返回帧信号矩阵


def pretune(co, data):
    # 实现对信号预加重，co为预加重系数，data为加重对象,一维数组.
    size_data = len(data)
    ad_signal = np.zeros(size_data)
    ad_signal[0] = data[0]
    # print(size_data)
    for i in range(1, size_data, 1):
        ad_signal[i] = data[i] - co * data[i - 1]  # 取矩阵中的数值用方括号
    return ad_signal


def dsilence(data, alfa, samplerate):
    print('----start desilence----')
    # 去除data的静音区，alfa为能量门限值
    tdata = pretune(0.955, data)
    edata = tdata / abs(tdata).max()  # 对语音进行归一化

    # frame_length分帧长度,hop_length帧偏移
    frame_length = int(50 * samplerate / 1000)  # 50ms帧长
    hop_length = frame_length
    winfunc = choose_windows('Hanning', frame_length)
    frames, nf = enframe(edata, frame_length, hop_length, winfunc)
    if nf != 1:
        frames = frames.T

        # 要以分割得到的帧数作为row
        row = frames.shape[1]  # 帧数
        col = frames.shape[0]  # 帧长

        print('帧数', frames.shape)
        Energy = np.zeros((1, row))

        # 短时能量函数
        for i in range(0, row):
            Energy[0, i] = np.sum(abs(frames[:, i] * frames[:, i]), 0)  # 不同分帧函数这里要换

        Ave_Energy = Energy.sum() / row
        Delete = np.zeros((1, row))

        # Delete(i)=1 表示第i帧为清音帧

        for i in range(0, row):
            if Energy[0, i] < Ave_Energy * alfa:
                Delete[0, i] = 1

        # 保存去静音的数据
        ds_data = np.zeros((frame_length * int(row - Delete.sum())))

        begin = 0
        for i in range(0, row - 1):
            if Delete[0, i] == 0:
                for j in range(0, frame_length, 1):
                    ds_data[begin * frame_length + j] = edata[i * hop_length + j]
                begin = begin + 1
        print('Numberofslices:', begin)
        print('ds_data:', ds_data.shape)
        ifdata = 1
        return ds_data, ifdata
    else:
        ifdata = 0
        return None, ifdata


def audioread(file_path):
    # Load audio file at its native sampling rate
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, samplerate, nframes = params[:4]  # nframes就是点数
    print("read audio dimension", nchannels, sampwidth, samplerate, nframes)
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    if waveData.size == 0:
        data = "empty_file"
        samplerate = -1
        return data,samplerate
    else:
        data = waveData[0, :]
        print("read audio size", data.shape)
        f.close()
        return data, samplerate


def writewav(audiodata, samplerate, audiopath):
    outData = audiodata  # 待写入wav的数据，这里仍然取waveData数据
    print("write audio size", outData.shape)
    print("max value", outData.max())
    outwave = wave.open(audiopath, 'wb')  # 定义存储路径以及文件名
    nchannels = 1
    sampwidth = 2  # 和数据存储的位数有关
    fs = samplerate
    data_size = len(outData)
    framerate = int(fs)
    nframes = data_size
    print("write nframes", nframes)

    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    for v in outData:
        if not np.isnan(v):
            outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # outData:16位转化为二进制，-32767~32767，注意不要溢出

    outwave.close()


def slicewav(audiofilespath, frame_time, overlap_rate):  # 保存剪切的音频'E:/基金/fastchirplet-master/fastchirplet-master/audio/'

    # 音频文件路径，帧长时间单位，帧的重叠率稍微大些
    filename = os.path.basename(audiofilespath)  # get filename
    print(filename)

    if audiofilespath.endswith(".wav"):
        # Load audio file at its native sampling rate
        data, sr = audioread(audiofilespath)
        if data != "empty_file":
            print("slice read audio size", data.shape, sr)
            # frame_time=500 #改变分割的长度，单位ms
            frame_length = int(sr * frame_time / 1000)
            hop_length = int((1 - overlap_rate) * frame_length)
            # 去除静音
            ds_data, ifdata = dsilence(data, 0.4, sr)  # 0.5是无声阈值，越高留下越少
            if ifdata != 0:
                # 生成新文件
                directory_in_str = os.path.dirname(audiofilespath)
                newaudiopath = directory_in_str + '/new/'
                print("filenew:", newaudiopath)
                if not os.path.exists(newaudiopath):
                    print("Can't find new files!")
                    os.makedirs(newaudiopath)
                winfunc = choose_windows('Hanning', frame_length)
                frames, nf = enframe(ds_data, frame_length, hop_length, winfunc)  # (帧数，帧长)
                if nf != 1:
                    frames = frames.T  # (帧长，帧数)
                    print("----create new files!----")
                    for i in range(0, frames.shape[1], 1):
                        file_path_new = os.path.join(newaudiopath,
                                                     filename[:-4] + str('-') + str(i) + str('.wav'))  # [:-3]指从后往前减3个

                        writewav(frames[:, i], sr, file_path_new)
                else:
                    return 0
            else:
                return 0


def compute_spectrogram(signal, sample_rate):
    """
        compute spectrogram from signal
        :param signal:
        :return: matrice representing the spectrum, frequencies corresponding and times
    """
    [spectrum, freqs, times] = plt.mlab.specgram(signal, NFFT=512, Fs=sample_rate,
                                                 noverlap=256, window=np.hamming(512))
    spectrum = 10. * np.log10(spectrum)

    return [spectrum, freqs, times]


def plotchirplet(chirps, audiopath):
    data, sr = audioread(audiopath)

    figure, axarr = plt.subplots(1, sharex=False)
    figure.set_size_inches(4, 4)
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    tabfinal = list(reversed(chirps))

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)
    mxf = freqs[index_frequency]
    print("----plot chirp----")

    axarr.matshow(tabfinal, cmap='Greens_r', origin='lower', extent=(0, times[-1], freqs[0], mxf), aspect='auto')

    plt.axis('off')
    plt.savefig(r"E:\Dataset\chirplet\negative\\" + audiopath[audiopath.rfind("\\")+1: -4] + str('ch') + str('.jpg'), dpi=56)
    audiopath[audiopath.rfind("\\")]
    plt.close('all')


def ge_graph(pathfile):
    chirplet = []
    print("chpath-file", pathfile)
    for root, dirs, files in os.walk(pathfile):

        for file in files:

            if file.endswith(".wav"):
                chirplet.append(os.path.join(root, file))

    for file in chirplet:
        data, sr = audioread(file)
        print("ch-file", file)
        ch1 = ch.FCT(sample_rate=sr)
        if data != "empty_file":
            chirps = ch1.compute(data)
            plotchirplet(chirps, file)


dir_path = r"E:\Dataset\negative"
file_list = os.listdir(dir_path)
for file in file_list:
    slicewav(os.path.join(dir_path, file), 1000, 0.5)

ge_graph(r"E:\Dataset\negative\new")
print("=========处理完成========")
