import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import matplotlib.pyplot as plt

from data import create_dataset
from model import EndToEnd
from callbacks import DisplaySentence

train_metadata_path = '../dataset/sv-SE/train.tsv'
test_metadata_path = '../dataset/sv-SE/test.tsv'
train_metadata = pd.read_csv(train_metadata_path, sep='\t')

train_dataset, vocab = create_dataset(train_metadata, base_dir="../dataset/sv-SE/clips/new/")

batch_size = 32
lr = 2e-4
patch_width = 32
model = EndToEnd(patch_width, 128)

optimizer = keras.optimizers.Adam(learning_rate=lr)
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),

train_dataset = train_dataset.batch(batch_size)
test_batch = list(train_dataset.take(1))[0][:8]

disp_callback = DisplaySentence(test_batch, vocab)

model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=True,
    metrics=['acc'],
)

# Make sure to adapt the vectorizer to the text corpus
history = model.fit(
    train_dataset,
    epochs=100,
    callbacks=[disp_callback],
)

