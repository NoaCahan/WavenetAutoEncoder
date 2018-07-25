import os
import glob
import librosa
import numpy as np

def mu_law_encode(audio, quantization_channels = 256):
    '''
    Arguments:
        audio: type(np.array), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        np.array of shape(sequence_length)
    Return:
        np.array with each element ranging from 0 to 255
        The size of return array is the same as input tensor
    '''
    mu = quantization_channels - 1
    safe_audio_abs = np.abs(np.clip(audio, -1.0, 1.0))
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    encoded = (signal + 1) / 2 * mu + 0.5
    return encoded.astype(np.int32)

def main(audio_dir, suffix='.wav'):
    pattern = audio_dir + '*' + suffix
    file_list = glob.glob(pattern)
    audio_list = []
    for item in file_list:
        audio = librosa.load(item, sr=16000, mono=True)
        audio = audio[0]
        encoded = mu_law_encode(audio)
        audio_list.append(encoded)
    output = open(audio_dir + "np_audio.pkl", 'wb')
    pickle.dump(audio_list, output)

def wav_to_numpy(audio_dir, suffix='.wav'):
    pattern = audio_dir + '*' + suffix
    file_list = glob.glob(pattern)
    audio_list = []
    for item in file_list:
        print(item)
        audio = librosa.load(item, sr=16000, mono=True)
        audio = audio[0]
        encoded = mu_law_encode(audio)
        audio_list.append(encoded)
    return audio_list
    
def wav_to_ndarray(audio_dir, ndarray_dir):
    pattern = audio_dir + '*' + '.wav'
    file_list = glob.glob(pattern)
    #audio_list = []
    for item in file_list:
        item_ndname = ndarray_dir + os.path.splitext(os.path.basename(item))[0] + '.npy'
        audio = librosa.load(item, sr=16000, mono=True)
        audio = audio[0]
        encoded = mu_law_encode(audio)
        np.save(item_ndname , encoded)
    #    audio_list.append(encoded)

def data_list(audio_dir, suffix='.npy'):
    print("starting to gather data as list")
    pattern = audio_dir + '*' + suffix
    file_list = glob.glob(pattern)
    audio_list = []
    for item in file_list:
        audio = np.load(item)
        audio_list.append(audio)
    return audio_list

if __name__ == '__main__':
    wav_to_ndarray('./data/fma_small_wav/', './data/fma_small_ndarray/')