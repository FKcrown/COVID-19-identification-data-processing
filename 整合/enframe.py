import numpy as np
import librosa
import scipy.signal as sg
import os
import soundfile as sf


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def pretune(co, data):
    # 实现对信号预加重，co为预加重系数，data为加重对象,一维数组.
    size_data = len(data)
    ad_signal = np.zeros(size_data)
    ad_signal[0] = data[0]
    # print(size_data)
    for i in range(1, size_data, 1):
        ad_signal[i] = data[i] - co * data[i - 1]  # 取矩阵中的数值用方括号
    return ad_signal


def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    winfunc窗函数winfunc = signal.hamming(nw)
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                         (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵


def split_audio(file_path, frame_time, overlap_rate):
    data, sr = librosa.load(file_path, sr=None)
    frame_length = int(sr * frame_time / 1000)
    hop_length = int((1 - overlap_rate) * frame_length)

    tdata = pretune(0.955, data)
    '''plt.plot(tdata)
    plt.ylabel('tdata')
    plt.show()'''

    winfunc = sg.hamming(frame_length)
    frames = enframe(tdata, frame_length, hop_length, winfunc)
    # frames = librosa.util.frame(tdata, frame_length, hop_length).T
    col = frames.shape[1]  # 1591#不同分帧函数这里要换
    row = frames.shape[0]  # 50
    # print('帧数', row, col)

    new_audio_path = os.path.join(file_path[:file_path.rfind("\\")], "new")
    filename = file_path.split('\\')[-1]

    create_dir_not_exist(new_audio_path)

    for i in range(0, row, 1):
        file_path_new = os.path.join(new_audio_path,
                                     filename[:-4] + str('-') + str(i) + str('.wav'))  # [:-3]指从后往前减3个
        sf.write(file_path_new, frames[i, :], sr)


print("切分完成")
