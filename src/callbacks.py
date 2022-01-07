import tensorflow as tf
from tensorflow import keras

class DisplaySentence(keras.callbacks.Callback):
    def __init__(self, batch, vocab):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            vocab: A vocabulary
        """
        self.batch = batch
        self.vocab = vocab
        self.start_idx = vocab.stoi[vocab.BOS]
        self.end_idx = vocab.stoi[vocab.EOS]

    def on_epoch_end(self, epoch, logs=None):
        source, target = self.batch
        batch_size = source.shape[0]
        preds = self.model.translate(source, 156, self.start_idx)
        preds = preds.numpy()
        print("")
        for i in range(batch_size):
            target_text = self.vocab.decode_tokens(target[i])
            prediction = self.vocab.decode_tokens(preds[i])
            print(f"target:     {target_text}")
            print(f"prediction: {prediction}\n")
