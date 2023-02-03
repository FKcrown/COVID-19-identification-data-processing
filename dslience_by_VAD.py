import webrtcvad as webrtcvad
from scipy.io import wavfile


def reduce_silence_by_VAD(input_file, out_file):
    sample_rate, samples = wavfile.read(input_file)
    vad = webrtcvad.Vad()
    vad.set_mode(3)

    raw_samples = struct.pack("%dh" % len(samples), *samples)

    window_duration = 0.03 # duration in seconds 0.03
    samples_per_window = int(window_duration * sample_rate + 0.3)
    bytes_per_sample = 2

    segments = []
    try:
        for start in range(0, len(samples), samples_per_window):
            stop = min(start + samples_per_window, len(samples))
            is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample],
                                  sample_rate = sample_rate)
            segments.append(dict(
                start = start,
                stop = stop,
                is_speech = is_speech))
    except:
        print(input_file+' sent error when reduce silence')
    speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
    write(out_file, sample_rate, speech_samples)