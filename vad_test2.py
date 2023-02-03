from vad import VoiceActivityDetector
import wave
import numpy as np

if __name__ == "__main__":
    load_file = r"C:\Users\Lollipop\Desktop\语音数据测试\test\AHvqxmTD_cough.wav"
    save_file = r"C:\Users\Lollipop\Desktop\语音数据测试\test\output\AHvqxmTD_cough.wav"
    # 获取vad分割节点
    v = VoiceActivityDetector(load_file)
    raw_detection = v.detect_speech()
    speech_labels, point_labels = v.convert_windows_to_readible_labels(raw_detection)
    if len(point_labels) != 0:
        # 根据节点音频分割并连接
        data = v.data
        cut_data = []
        Fs = v.rate
        for start, end in point_labels:
            cut_data.extend(data[int(start):int(end)])

        # 保存音频
        f = wave.open(save_file, 'w')
        nframes = len(cut_data)
        f.setparams((1, 2, Fs, nframes, 'NONE', 'NONE'))  # 声道，字节数，采样频率，*，*
        wavdata = np.array(cut_data)
        wavdata = wavdata.astype(np.int16)
        f.writeframes(wavdata)  # outData
        f.close()
