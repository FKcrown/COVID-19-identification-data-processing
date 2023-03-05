import csv
import os
from warnings import simplefilter

import librosa
import librosa.util
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import soundfile as sf
from tqdm import tqdm

import chirplet as ch
from enframe import enframe
from enframe import pretune
from vad import filter

simplefilter(action='ignore', category=FutureWarning)


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


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


def vad(file_dir, resample_sr):
    resample_dir = os.path.join(file_dir, "resample")
    vad_dir = os.path.join(file_dir, "vad")
    # 创建resample重采样文件夹和vad静音删除文件夹
    create_dir_not_exist(resample_dir)
    create_dir_not_exist(vad_dir)

    file_list = os.listdir(file_dir)
    for file_name in tqdm(file_list, desc="静音删除", ncols=80, colour='green'):
        if file_name.endswith(".wav"):
            print(file_name)
            file_path = os.path.join(file_dir, file_name)

            resample_path = os.path.join(resample_dir, file_name[:-4] + "-{}K".format(resample_sr / 1000) + ".wav")

            file, sr = librosa.load(file_path, sr=None)
            # 检测该音频是否为静音
            if not abs(max(file.min(), file.max())) < 0.005:
                # 音频重采样至resample_sr并存放到resample_dir
                file_resample = librosa.resample(file, sr, resample_sr)
                sf.write(resample_path, file_resample, resample_sr)
                # 对采样后的音频进行vad静音删除并存放到vad_dir
                filter(resample_path, vad_dir, expand=False)
                os.remove(resample_path)  # 删除resample_dir里的文件
            else:
                print("{}为空音频".format(file_path))
                with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(file_path)  # 空音频路径写入csv文件


def split_audio_files(audio_dir, frame_time, overlap_rate, check_duration=True):
    """
    对指定目录下的音频文件进行处理，将长度大于等于 frame_time 的音频文件切分至 frame_time 长度，
    重叠率为 overlap_rate。
    :param audio_dir: 音频文件所在目录
    :param frame_time: 指定切分的音频长度，单位为毫秒
    :param overlap_rate: 指定音频的重叠率，取值为 0-1 之间的小数
    :param check_duration: 是否检查音频长度是否大于等于 frame_time，当值为 False 时，直接切分音频文件
    """
    for file in tqdm(os.listdir(audio_dir), desc="音频切分", ncols=80, colour='blue'):
        if file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, file)
            # 如果check_duration为True，则检查音频长度是否大于等于frame_time
            if check_duration:
                duration = librosa.get_duration(filename=audio_path)
                if duration >= frame_time / 1000:
                    print(audio_path)
                    # 切分音频至 frame_time 长度，重叠率为 overlap_rate
                    split_audio(audio_path, frame_time, overlap_rate)
                else:
                    print("{} 音频长度小于 {}s，不处理".format(audio_path, frame_time / 1000))
            # 如果check_duration为False，则直接切分音频文件
            else:
                print(audio_path)
                # 切分音频至 frame_time 长度，重叠率为 overlap_rate
                split_audio(audio_path, frame_time, overlap_rate)


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

    figure.savefig(audiopath[:-4] + str('.jpg'), dpi=100)
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

    melspec = librosa.feature.mfcc(data, sr, n_mfcc=32, n_fft=1024, hop_length=512)  # 计算mel倒谱

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

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\MFCC')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'),
                   dpi=56)
    # plt.show()

    plt.close('all')


def plotlogmelspec(audiopath):
    print(audiopath)

    data, sr = librosa.load(audiopath, sr=None)

    figure, axarr = plt.subplots(1, sharex=False)

    figure.set_size_inches(4, 4)  # 输出图大小和时间长度成正比50ms一幅图
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)

    max_frequency = freqs[index_frequency]

    melspec = librosa.feature.melspectrogram(data, sr, n_fft=1024, hop_length=512, n_mels=128,
                                             fmax=8000)
    logmelspec = librosa.power_to_db(melspec)
    # print('mel',np.shape(melspec))

    axarr.matshow(logmelspec[1:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], 0, 8000),
                  aspect='auto')

    plt.axis('off')

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\logMel')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'),
                   dpi=56)
    # plt.show()

    plt.close('all')


def plotTFDF(audiopath):
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

    [spectrum, freqs, times] = plt.mlab.specgram(data, NFFT=1024, Fs=sr,
                                                 noverlap=512, window=np.hamming(1024))
    melspec = librosa.feature.melspectrogram(data, sr, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)

    TFDF = compute_TFDF(logmelspec)

    index_frequency = np.argmax(freqs)
    max_frequency = freqs[index_frequency]

    axarr.matshow(TFDF[0:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], 0, 8000),
                  aspect='auto')

    plt.axis('off')

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\TFDF')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'),
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

    ch_folder = audiopath[: audiopath.rfind("\\")] + str(r'\chirplet')

    create_dir_not_exist(ch_folder)
    figure.savefig(ch_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'), dpi=56)
    # plt.show()

    plt.close('all')


def plotchTFDF(chirps, audiopath):
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
    temp = list(reversed(chirps))
    tabfinal = np.array(temp).reshape(len(temp), -1)
    TFDF = compute_TFDF(tabfinal)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)

    index_frequency = np.argmax(freqs)
    mxf = freqs[index_frequency]
    # print("最大频率",mxf)

    axarr.matshow(TFDF, cmap='jet',
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

    ch_folder = audiopath[: audiopath.rfind("\\")] + str(r'\chTFDF')

    create_dir_not_exist(ch_folder)
    figure.savefig(ch_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'), dpi=56)
    # plt.show()

    plt.close('all')


def ge_graph(pathfile):
    chirplet = []

    for root, dirs, files in os.walk(pathfile):

        for file in files:

            if file.endswith(".wav"):
                chirplet.append(os.path.join(root, file))

    for file in tqdm(chirplet, desc="绘制谱图", ncols=80):
        # start_time = time.time()
        data, sr = librosa.load(file, sr=None)
        # data=data/abs(data).max()#对语音进行归一化
        ch1 = ch.FCT(sample_rate=sr)
        chirps = ch1.compute(data)
        # print(chirps)
        # plotchirplet(chirps, file)
        # plotchTFDF(chirps, file)
        # plotmelspec(file)
        # plotlogmelspec(file)
        plotTFDF(file)
        # plotsft(file)
        # joblib.dump(chirps, file[:-3] + 'jl')


folder_path = r"F:\Database\Audios\Track1+CoughVid\positive"
resample_sr = 16000  # 音频重采样频率，单位：Hz
frame_time = 2000  # 指定切分的音频长度，单位：ms
overlap_rate = 0.5  # 指定切分音频的重叠率，取值为0-1的小数
# vad(folder_path, resample_sr)
# split_audio_files(os.path.join(folder_path, 'vad'), frame_time, overlap_rate)
ge_graph(os.path.join(folder_path, "vad", 'new'))
print("finish!!")
