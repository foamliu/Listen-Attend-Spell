import librosa
import librosa.display
import matplotlib.pyplot as plt
from python_speech_features import mfcc

def plot_wave(path):
    """
    Args:
        path: Path to the audio file we want to plot
    """
    samples, sample_rate = librosa.load(path, mono=True, sr=None)
    plt.figure(figsize=[15, 5])
    librosa.display.waveplot(samples, sr=sample_rate)
    plt.show()


def plot_melspectogram(path, n_mels=128):
    """
    Args:
        path: The path to to the audiofile we want to plot.
    """
    samples, sample_rate = librosa.load(path, mono=True, sr=None)
    plt.figure(figsize=[20, 5])
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S)
    plt.show()



def audioToInputVector(audio_filename, numcep, numcontext):
    """
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    Borrowed from Mozilla's Deep Speech and slightly modified.
    https://github.com/mozilla/DeepSpeech
    """

    audio, fs = librosa.load(audio_filename)

    # # Get mfcc coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep, nfft=551)
    # features = librosa.feature.mfcc(y=audio,
    #                                 sr=fs,
    #                                 n_fft=551,
    #                                 n_mfcc=numcep).T

    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]

    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)

    features = np.concatenate((empty_context, features, empty_context))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2 * numcontext + 1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    # Whiten inputs (TODO: Should we whiten?)
    # Copy the strided array so that we can write to it safely
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    # Return results
    return train_inputs