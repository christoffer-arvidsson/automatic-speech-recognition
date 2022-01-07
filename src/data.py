import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from vocab import Vocab

def read_audio(path):
    audio = tfio.audio.AudioIOTensor(path, dtype=tf.float32)
    audio_tensor = tf.squeeze(audio.to_tensor(), axis=[-1])

    return audio_tensor

def apply_trim(signal):
    position = tfio.audio.trim(signal, axis=0, epsilon=0.1)
    start = position[0]
    stop = position[1]

    processed = signal[start:stop]

    return processed

def apply_preemphasis_filter(signal):
    """Amplify high frequencies to balance magnitudes of higher
    frequencies and improve signal to noise ratio."""
    pre_emphasis = 0.97
    emph_signal = tf.concat([signal[:1], signal[:1] - pre_emphasis * signal[:-1]], 0)

    return emph_signal

def apply_spectrogram(signal):
    spec = tfio.audio.spectrogram(signal, nfft=512, window=256, stride=128)
    # log_spec = tf.math.log(spec)

    return spec

def apply_log_mel_spec(signal):
    w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=100, num_spectrogram_bins=257, sample_rate=32000)
    return tf.math.log(tf.matmul(signal, w) + 1e-6)

def apply_mfcc(log_mel_spec):
    # We keep the first 2-13 for speech because higher represent fast
    # changes in filter bank coefficients, which don't contribute to ASR
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spec)[:,1:13]

    return mfcc

def apply_pad(signal, pad_len):
    # to_pad = max(0, pad_len - signal.shape[0])
    to_pad = pad_len
    paddings = tf.constant([[0, to_pad]])
    x = tf.pad(signal, paddings, "CONSTANT")[:to_pad]

    return x

def apply_normalization(spec):
    means = tf.math.reduce_mean(spec, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spec, 1, keepdims=True)
    return (spec - means) / stddevs
    
def preprocess_dataset(dataset):
    """Pipeline for preprocessing the dataset of signals and word encodings."""
    new = (dataset
           .map(map_func=lambda x,y: (apply_trim(x), y))
           .map(map_func=lambda x,y: (apply_preemphasis_filter(x), y))
           .map(map_func=lambda x,y: (apply_pad(x, 100000), y))
           .map(map_func=lambda x,y: (apply_spectrogram(x), y))
           .map(map_func=lambda x,y: (apply_log_mel_spec(x), y))
           # .map(map_func=lambda x,y: (apply_mfcc(x), y))
           )

    return new
    
def create_dataset(metadata, vocab=None, base_dir=""):
    """Create a dataset from a metadata pandas frame. If vocab is not supplied, build a new vocabulary."""
    # Encode target sentences into word labels
    sentences = metadata.sentence
    vocab = Vocab()
    vocab.build(sentences)

    labels = vocab.encode_docs(sentences)

    paths = metadata.path
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(map_func=lambda x,y: (read_audio(base_dir + x), y))
    pre_dataset = preprocess_dataset(dataset).shuffle(1024)

    return pre_dataset, vocab

