import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data import create_dataset
from model import EndToEnd
from callbacks import checkpoint_cb
from jiwer import cer, wer

batch_size = 64
patch_width = 32
lr = 2e-4
checkpoint_filepath = "../checkpoint/nlp_transformer/"
train_metadata_path = "../dataset/sv-SE/validated.tsv"
test_metadata_path = "../dataset/sv-SE/test.tsv"
train_metadata = pd.read_csv(train_metadata_path, sep="\t")
test_metadata = pd.read_csv(test_metadata_path, sep="\t")

train_dataset, vocab = create_dataset(
    train_metadata, base_dir="../dataset/sv-SE/clips/new/"
)
test_dataset, _ = create_dataset(
    test_metadata, base_dir="../dataset/sv-SE/clips/new/", vocab=vocab
)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_batch = list(test_dataset.take(1))[0]  # For training feedback

model = EndToEnd(patch_width, 128)

optimizer = keras.optimizers.Adam(learning_rate=lr)
loss = (tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),)

# Load checkpoint
checkpoint_cb = checkpoint_cb(checkpoint_filepath)

model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=True,
)

model.load_weights(checkpoint_filepath)

max_target_length = 100
average_cer = 0
average_wer = 0
num_batches = len(test_dataset)
for i, batch in enumerate(test_dataset):
    source, target = batch
    target = vocab.decode_docs(target.numpy())
    pred = model.translate(source, max_target_length, vocab.stoi[vocab.BOS])
    pred = vocab.decode_docs(pred.numpy())
    cer_score = cer(target, pred)
    wer_score = wer(target, pred)
    average_cer += cer_score
    average_wer += wer_score
    print(
        pred[0], target[0], f"WER: {average_wer / (i+1)}", f"CER: {average_cer / (i+1)}"
    )
