import os
import glob
from pydub import AudioSegment

# Path where the mp3s are located
mp3_dir = './fma_small'
wav_dir = './fma_small_wav/'
extension = ('*.mp3')

#os.chdir(mp3_dir)
for root, dirs, files in os.walk(mp3_dir):
    for file in files:
        if os.path.splitext(os.path.basename(file))[1] == '.mp3' : 
            full = os.path.join(root, file)
            wav_filename = os.path.splitext(os.path.basename(file))[0] + '.wav'
            wav_path = wav_dir + wav_filename
            print(full)
            try:
                sound = AudioSegment.from_mp3(full)
                sound.export(wav_path, format='wav')
            except:
                pass
                