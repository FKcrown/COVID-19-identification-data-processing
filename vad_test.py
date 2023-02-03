from vad import VoiceActivityDetector
import argparse
import json


def save_to_file(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    inputfile = r"C:\Users\Lollipop\Desktop\语音数据测试\test\AHvqxmTD_cough.wav"
    outputfile = r"C:\Users\Lollipop\Desktop\语音数据测试\test\output\AHvqxmTD_cough.wav"

    v = VoiceActivityDetector(inputfile)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)

    save_to_file(speech_labels, outputfile)

