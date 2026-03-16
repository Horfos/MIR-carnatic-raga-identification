import librosa 
import config
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    return y
def compute_logmel(y):
    stft= librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    spectrogram= np.abs(stft)    

    mel = librosa.feature.melspectrogram(
        S=spectrogram**2,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def pad_or_crop_logmel(log_mel, expected_frames):

    if log_mel.shape[1] < expected_frames:
        pad_width = expected_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    elif log_mel.shape[1] > expected_frames:
        log_mel = log_mel[:, :expected_frames]

    return log_mel

def extract_logmel(audio_path):
    '''Complete feature extraction pipeline'''
    y = load_audio(audio_path)
    log_mel = compute_logmel(y)

    expected_frames = int(config.CLIP_DURATION * config.SAMPLE_RATE / config.HOP_LENGTH)
    log_mel = pad_or_crop_logmel(log_mel, expected_frames)

    return log_mel

def plot_logmel(log_mel):
    plt.figure(figsize=(10,4))
    
    librosa.display.specshow(
        log_mel,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        x_axis='time',
        y_axis='mel'
    )

    plt.colorbar(format='%+2.0f dB')
    plt.title("Log Mel Spectrogram")
    plt.show()