import numpy as np
from 整合 import chirplet as ch
import matplotlib.pyplot as plt
import librosa.util
import librosa
import joblib
import os
import scipy.signal as sg
from os.path import exists
import soundfile as sf


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


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
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile (np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵


def pretune(co, data):
    # 实现对信号预加重，co为预加重系数，data为加重对象,一维数组.
    size_data = len(data)
    ad_signal = np.zeros(size_data)
    ad_signal[0] = data[0]
    # print(size_data)
    for i in range(1, size_data, 1):
        ad_signal[i] = data[i] - co * data[i - 1]  # 取矩阵中的数值用方括号
    return ad_signal


def dsilence(data, alfa, frame_length, hop_length):
    # 去除data的静音区，alfa为能量门限值，frame_length分帧长度,, hop_length帧偏移
    tdata = pretune(0.955, data)
    '''plt.plot(tdata)
    plt.ylabel('tdata')
    plt.show()'''
    edata = tdata / abs(tdata).max()  # 对语音进行归一化

    winfunc = sg.hamming(frame_length)
    frames = enframe(tdata, frame_length, hop_length, winfunc)
    # frames = librosa.util.frame(edata, frame_length, hop_length)  # 这个更接近matlab结果,不要重叠
    '''librosa.util.frame:
    Parameters:	
    y : np.ndarray [shape=(n,)]

    Time series to frame. Must be one-dimensional and contiguous in memory.

    frame_length : int > 0 [scalar]

    Length of the frame in samples

    hop_length : int > 0 [scalar]

    Number of samples to hop between frames

    Returns:	
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]'''
    # librosa.util.frame和enframe生成的frames的维数正好是反的，要以分割得到的帧数作为row
    col = frames.shape[1]  # 1591#不同分帧函数这里要换
    row = frames.shape[0]  # 50

    # print('帧数',row,col)
    Energy = np.zeros((1, row))

    # 短时能量函数
    for i in range(0, row - 1):
        Energy[0, i] = np.sum(abs(frames[:, i] * frames[:, i]), 0)  # 不同分帧函数这里要换

    MAX = Energy.max()
    Ave_Energy = Energy.sum() / row
    Delete = np.zeros((1, row))

    # Delete(i)=1 表示第i帧为清音帧

    for i in range(0, row - 1):
        if Energy[0, i] < Ave_Energy * alfa:
            Delete[0, i] = 1

    # 暂存每一帧的数据
    ds_new = np.zeros((frame_length))
    # 保存每一帧的数据，横为帧数，纵为帧长
    ds_data = np.zeros((frame_length, int(row - Delete.sum())))

    begin = 0
    for i in range(0, row - 1):
        if Delete[0, i] == 0:
            for j in range(0, frame_length - 1, 1):
                # New[begin*hop_length+j]=edata[i*hop_length+j]
                ds_new[j] = edata[i * hop_length + j]
            ds_data[:, begin] = ds_new
            begin = begin + 1
    print('Numberofslices:', begin)
    return ds_data, edata


def slicewav(audiofilespath, frame_time, overlap_rate):  # 保存剪切的音频'E:/基金/fastchirplet-master/fastchirplet-master/audio/'

    # 音频文件路径，帧长时间单位，帧的重叠率
    directory_in_str = audiofilespath

    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_in_str, filename)
            print(file_path)
            # Load in librosa's example audio file at its native sampling rate
            data, sr = librosa.load(file_path, sr=None)
            '''plt.plot(data)
            plt.ylabel('data')
            plt.show()'''
            # frame_time=500 #改变分割的长度，单位ms
            frame_length = int(sr * frame_time / 1000)
            hop_length = int((1 - overlap_rate) * frame_length)
            # 去除静音
            ds_data, edata = dsilence(data, 0.7, frame_length, hop_length)  # 0.6是无声阈值，越高留下越小
            '''plt.plot(newdata)
            plt.ylabel('newdata')
            plt.show()'''
            # 生成新文件
            '''file_path_new = os.path.join(directory_in_str+'new/', filename[:-4]+str('-')+str('source')+str('.wav'))#[:-3]指从后往前减3个
            librosa.output.write_wav(file_path_new, edata, sr)'''
            newaudiopath = directory_in_str + '/new/'

            if not exists(newaudiopath):
                print("Can't find original files!")
                os.makedirs(newaudiopath)

            for i in range(0, ds_data.shape[1] - 1, 1):
                file_path_new = os.path.join(newaudiopath,
                                             filename[:-4] + str('-') + str(i) + str('.wav'))  # [:-3]指从后往前减3个
                sf.write(file_path_new, ds_data[:, i], sr)


# slicewav('E:/基金/fastchirplet-master/fastchirplet-master/audio2/')
# 存为mat文件，sio.savemat('saveddata.mat', {'dda': ddata})

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
    # axarr.axes.xaxis.set_ticks_position('bottom')
    # axarr.set_ylabel("Frequency in Hz")
    # axarr.xaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)
    # axarr.yaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)

    # axarr.set_title('melspectrogram')

    # figure.tight_layout()

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\mel')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('mel') + str('.jpg'),
                   dpi=56)
    # plt.show()

    plt.close('all')


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

    plt.axis('off')
    # axarr.axes.xaxis.set_ticks_position('bottom')
    # axarr.set_ylabel("Frequency in Hz")
    # axarr.xaxis.grid(which='major', color='Black',
    #  linestyle='-', linewidth=0.25)
    # axarr.yaxis.grid(which='major', color='Black',
    #  linestyle='-', linewidth=0.25)

    # axarr.set_title('chirplet')

    # figure.tight_layout()

    ch_folder = audiopath[: audiopath.rfind("\\")] + str(r'\mel')

    create_dir_not_exist(ch_folder)
    figure.savefig(ch_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('ch') + str('.jpg'), dpi=56)
    # plt.show()

    plt.close('all')


def ge_graph(pathfile):
    chirplet = []

    for root, dirs, files in os.walk(pathfile):

        for file in files:

            if file.endswith(".wav"):
                chirplet.append(os.path.join(root, file))

    for file in chirplet:
        # start_time = time.time()
        data, sr = librosa.load(file, sr=None)
        # data=data/abs(data).max()#对语音进行归一化
        ch1 = ch.FCT(sample_rate=sr)
        chirps = ch1.compute(data)
        # print(chirps)
        # plotchirplet(chirps, file)
        plotmelspec(file)
        # plotsft(file)
        joblib.dump(chirps, file[:-3] + 'jl')


# def test(data_dir):
# if __name__ == '__main__':
#     start_time = time.time()
#
#     #data_dir ='E:/DLPROGRAM/fastchirplet-master/fastchirplet-master/bird10chirp/'
#     data_dir = "E:\DICOVA_Track1\Track1-classified"
#     contents = os.listdir(data_dir)  # return the file name
#     print(contents)
#     audiofiles = [each for each in contents if os.path.isdir(data_dir + each)]  # decide if a file
#     print(audiofiles)
#     for each in audiofiles:
#         ldata_dir=data_dir + each
#         slicewav(ldata_dir,1000,0.5)
#         ge_graph(ldata_dir+'/new/')
#     #ge_graph(pathfile)
#     end_time = time.time()
#     print("--- %s seconds ---" % (end_time - start_time))


dir_path = r"C:\Users\Lollipop\Desktop\语音数据测试\test\plot"
slicewav(dir_path, 1000, 0.5)
ge_graph(os.path.join(dir_path, 'new'))
print("====Finish!====")
