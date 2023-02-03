import librosa
import soundfile as sf
# to install librosa package
# > conda install -c conda-forge librosa

filename = r"C:\Users\Lollipop\Desktop\语音数据测试\test\CcOAENdR_cough.wav"
newFilename = r"C:\Users\Lollipop\Desktop\语音数据测试\test\output\CcOAENdR_cough-48K.wav"

y, sr = librosa.load(filename, sr=None)
y_48k = librosa.resample(y, sr, 48000)

sf.write(newFilename, y_48k, 48000)

