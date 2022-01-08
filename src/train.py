import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import matplotlib.pyplot as plt

from data import create_dataset, split_train_validation
from model import EndToEnd
from callbacks import DisplaySentence, checkpoint_cb

train_metadata_path = '../dataset/sv-SE/train.tsv'
test_metadata_path = '../dataset/sv-SE/test.tsv'
train_metadata = pd.read_csv(train_metadata_path, sep='\t')
test_metadata = pd.read_csv(test_metadata_path, sep='\t')

dataset, vocab = create_dataset(train_metadata, base_dir="../dataset/sv-SE/clips/new/")
test_dataset, _ = create_dataset(test_metadata, base_dir="../dataset/sv-SE/clips/new/", vocab=vocab)

train_dataset, val_dataset = split_train_validation(dataset, train_ratio=0.9)

batch_size = 64
lr = 2e-4
patience = 5
patch_width = 32
model = EndToEnd(patch_width, 128)

optimizer = keras.optimizers.Adam(learning_rate=lr)
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),

train_dataset = train_dataset.batch(batch_size).shuffle(1024).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE)
test_batch = list(test_dataset.take(1))[0] # For training feedback

# Callbacks
disp_callback = DisplaySentence(test_batch, vocab)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=patience, min_delta=0, restore_best_weights=True
)
checkpoint_cb = checkpoint_cb('/mnt/media/checkpoint/nlp_transformer/')

model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=True,
    metrics=['val_loss'],
)

# Make sure to adapt the vectorizer to the text corpus
history = model.fit(
    train_dataset,
    validation_data = val_dataset.batch(batch_size),
    epochs=100,
    callbacks=[disp_callback, early_stopping, checkpoint_cb],
)

