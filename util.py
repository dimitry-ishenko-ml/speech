import io
import json
import open_speech
import os
import tensorflow as tf

from open_speech import AUTOTUNE, sample_rate, dtype
from tensorflow import keras

####################
batch_size = 512

max_audio_size = 20 # 20s
max_label_size = 200 # 200 chars

frame_size = .025 # 25ms
frame_step = .010 # 10ms

fft_length = 1024

num_mel_bins = 80
lower_edge_hertz = 80
upper_edge_hertz = 7600

use_mfccs = True
num_mfccs = 13

use_frequency_mask = True
num_frequency_mask = 1
max_frequency_mask_width = 0.10 # 10% == max ~1 mfcc band

use_time_mask = True
num_time_mask = 2
max_time_mask_width = 0.05 # 5% == max 1 sec

####################
def init_data():
    global _max_audio_size, _max_label_meta_size
    _max_audio_size = max_audio_size * sample_rate
    _max_label_meta_size = max_label_size + 2 # see _deserialize for +2

    global _frame_size, _frame_step
    _frame_size = int(round(sample_rate * frame_size))
    _frame_step = int(round(sample_rate * frame_step))

    global _max_frame_steps
    _max_frame_steps = 1 + (_max_audio_size - _frame_size) // _frame_step

    global labels, _table
    labels = {
        uuid: open_speech.clean(label) for uuid, label in open_speech.labels.items()
    }
    _table = open_speech.lookup_table(labels)

    global alphabet, num_chars, _encoder, _decoder
    alphabet = set()
    for label in labels.values(): alphabet |= set(label)
    alphabet = sorted(alphabet) + ["∅"]
    num_chars = len(alphabet)

    _encoder = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer( keys=alphabet, values=range(num_chars) ),
        default_value=-1
    )
    _decoder = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer( keys=range(num_chars), values=alphabet ),
        default_value=""
    )

def get_input_shape():
    return (_max_frame_steps, num_mfccs if use_mfccs else num_mel_bins)

####################
def create_strategy(tpu_addr=None):
    if tpu_addr is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + tpu_addr)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver)

    else: return tf.distribute.MirroredStrategy()

####################
def _deserialize(example):
    uuid, audio = open_speech.parse_serial(example)
    label = _table.lookup(uuid)
    label = _encoder.lookup(tf.strings.bytes_split(label))

    # compute audio and label length, as they will be needed for ctc_loss();
    # this has to be done before calling padded_batch()
    steps = 1 + tf.math.maximum(0, tf.size(audio) - _frame_size) // _frame_step
    chars = tf.size(label)

    # prepend audio and label length in front of the label,
    # so we can sneak them in to the ctc_loss() function
    label_meta = tf.concat([ [steps], [chars], label ], axis=-1)

    return audio, label_meta

def _transform(audio, label_meta):
    stfts = tf.signal.stft(audio,
        frame_length=_frame_size, frame_step=_frame_step, fft_length=fft_length
    )
    specs = tf.abs(stfts)

    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins, num_spectrogram_bins=stfts.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz
    )
    mel_specs = tf.tensordot(specs, mel_weight_matrix, axes=1)
    log_mel_specs = tf.math.log(mel_specs + 1e-6)

    if use_mfccs:
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_specs)
        return mfccs[..., : num_mfccs], label_meta

    else: return log_mel_specs, label_meta

def _spec_augment(signal, label_meta):
    batch_size, num_steps, num_bins = signal.shape

    if use_frequency_mask:
        for _ in range(num_frequency_mask):
            max_width = int(max_frequency_mask_width * num_bins)
            f1 = tf.random.uniform([], 0, max_width, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, num_bins - f1, dtype=tf.int32)

            frequency_mask = tf.concat([
                tf.ones ([batch_size, num_steps, f0]),
                tf.zeros([batch_size, num_steps, f1]),
                tf.ones ([batch_size, num_steps, num_bins - f0 - f1]),
            ], axis=2)

            signal = signal * frequency_mask

    if use_time_mask:
        for _ in range(num_time_mask):
            max_width = int(max_time_mask_width * num_steps)
            t1 = tf.random.uniform([], 0, max_width, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, num_steps - t1, dtype=tf.int32)

            time_mask = tf.concat([
                tf.ones ([batch_size, t0, num_bins]),
                tf.zeros([batch_size, t1, num_bins]),
                tf.ones ([batch_size, num_steps - t0 - t1, num_bins]),
            ], axis=1)

            signal = signal * time_mask

    return signal, label_meta

def _get_dataset(dataset, prefetch, use_spec_augment):
    # serial -> (audio, label_meta)
    dataset = dataset.map(_deserialize, num_parallel_calls=AUTOTUNE)

    # discard long examples
    dataset = dataset.filter(lambda audio, label_meta:
        tf.size(audio) <= _max_audio_size and tf.size(label_meta) <= _max_label_meta_size
    )

    dataset = dataset.padded_batch(batch_size=batch_size,
        padded_shapes=( [_max_audio_size], [_max_label_meta_size] ),
        padding_values=( None, -1 ),
        drop_remainder=True
    )

    # (audio, label_meta) -> (log_mel_specs or mfccs, label_meta)
    dataset = dataset.map(_transform, num_parallel_calls=AUTOTUNE)

    if use_spec_augment:
        dataset = dataset.map(_spec_augment, num_parallel_calls=AUTOTUNE)

    if prefetch: dataset = dataset.prefetch(prefetch)

    return dataset

####################
def get_train_dataset(prefetch=None):
    return _get_dataset(open_speech.train_recordset,
        prefetch=prefetch, use_spec_augment=True
    )

def get_valid_dataset(prefetch=None):
    return _get_dataset(open_speech.valid_recordset,
        prefetch=prefetch, use_spec_augment=False
    )

def get_test_dataset(prefetch=None):
    return _get_dataset(open_speech.test_recordset,
        prefetch=prefetch, use_spec_augment=False
    )

####################
def ctc_loss(label_meta, y_pred):
    label_meta = tf.cast(label_meta, tf.int32)
    # extract audio and sentence length that were prepended to the label
    steps, chars, labels = label_meta[:, 0], label_meta[:, 1], label_meta[:, 2:]

    return tf.nn.ctc_loss(labels=labels, logits=y_pred,
        label_length=chars, logit_length=steps, logits_time_major=False, blank_index=-1
    )

def ctc_decode(y_pred, steps):
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    decoded, _ = tf.nn.ctc_greedy_decoder(inputs=y_pred, sequence_length=steps)
    return tf.cast(decoded[0], dtype=tf.int32)

def edit_distance(label_meta, y_pred):
    label_meta = tf.cast(label_meta, dtype=tf.int32)
    # extract audio and sentence length that were prepended to the label
    steps, chars, labels = label_meta[:, 0], label_meta[:, 1], label_meta[:, 2:]

    labels = keras.backend.ctc_label_dense_to_sparse(labels, chars)
    decoded = ctc_decode(y_pred=y_pred, steps=steps)

    return tf.edit_distance(hypothesis=decoded, truth=labels)
