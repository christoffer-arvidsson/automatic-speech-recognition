import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import matplotlib.pyplot as plt

from data import create_dataset
from model import EndToEnd

train_metadata_path = "../dataset/sv-SE/train.tsv"
train_metadata = pd.read_csv(train_metadata_path, sep="\t")

train_dataset, vocab = create_dataset(
    train_metadata, base_dir="../dataset/sv-SE/clips/new/"
)

batch_size = 8
lr = 0.01
patch_width = 16
model = EndToEnd(patch_width, 128)

train_dataset = train_dataset.batch(8)
data = list(train_dataset.take(1))
# out = model.call((xs, ys), True)

n_patches = 8
fig, axes = plt.subplots(1, n_patches, sharey=True)
for i in range(n_patches):
    axes[i].imshow(data[0][0][i].numpy(), origin="lower")

plt.show()
