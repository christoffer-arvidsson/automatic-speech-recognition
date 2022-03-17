import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json

from data import create_dataset
from model import EndToEnd
from callbacks import DisplaySentence, checkpoint_cb

checkpoint_filepath = "../checkpoint/nlp_transformer/"
train_metadata_path = "../dataset/sv-SE/validated.tsv"
val_metadata_path = "../dataset/sv-SE/dev.tsv"
test_metadata_path = "../dataset/sv-SE/test.tsv"
train_metadata = pd.read_csv(train_metadata_path, sep="\t")
val_metadata = pd.read_csv(val_metadata_path, sep="\t")
test_metadata = pd.read_csv(test_metadata_path, sep="\t")

train_dataset, vocab = create_dataset(
    train_metadata, base_dir="../dataset/sv-SE/clips/new/"
)
val_dataset, _ = create_dataset(
    val_metadata, base_dir="../dataset/sv-SE/clips/new/", vocab=vocab
)
test_dataset, _ = create_dataset(
    test_metadata, base_dir="../dataset/sv-SE/clips/new/", vocab=vocab
)

batch_size = 64
lr = 2e-4
patience = 10
patch_width = 32
model = EndToEnd(patch_width, vocab.size)

optimizer = keras.optimizers.Adam(learning_rate=lr)
loss = (tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),)

train_dataset = train_dataset.batch(batch_size).shuffle(1024).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE)
test_batch = list(test_dataset.take(1))[0]  # For training feedback

# Callbacks
disp_callback = DisplaySentence(test_batch, vocab)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, min_delta=0, restore_best_weights=True
)
checkpoint_cb = checkpoint_cb(checkpoint_filepath)

model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=True,
    metrics=["val_loss"],
)

# Load checkpoin
model.load_weights(checkpoint_filepath)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[disp_callback, early_stopping],
)

# Get the dictionary containing each metric and the loss for each epoch
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open("history.json", "w"))
