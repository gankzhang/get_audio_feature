import librosa
import os
import fnmatch
import param
import numpy as np

def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def find_files(directory, pattern='*.mp3'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files



def read_audios(receptive_field = 1024,sample_rate = 24000,sample_size = 10000,batch_size = 1,audio_dir = None):
    if audio_dir == None:
        audio_dir = param.get_audio_dir()
    files = find_files(audio_dir)
    total_audios = []
    audios = []
    audio_id = 0
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
#        audio = trim_silence(audio[:, 0], 0.3)
        audio = np.pad(audio, [[receptive_field, 0], [0, 0]],
                        'constant')
        if sample_size == 'all':
            total_audios.append(audio[:,:])
            assert batch_size == 1
            print('all')
            continue
        while len(audio) > receptive_field + sample_size:
            piece = audio[:(receptive_field +
                            sample_size), :]
            audios.append(piece)
            audio_id+=1
            if audio_id == batch_size:
                total_audios.append(audios)
                audios = []
                audio_id = 0
            audio = audio[(receptive_field +sample_size):,:]
#        total_audios.append(audios)
        print(filename)
    return total_audios


if __name__ == '__main__':
#    audio = read_audios()
#    audio = np.pad(audio, [[receptive_field, 0], [0, 0]],
#                    'constant')
    audios = read_audios(audio_dir = './test_reader',sample_size = 'all')
    print(0)