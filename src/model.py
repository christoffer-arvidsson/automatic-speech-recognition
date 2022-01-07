import tensorflow as tf
from tensorflow.keras import layers

class EndToEnd(tf.keras.Model):
    def __init__(self, patch_width, target_vocab_size):
        super().__init__()
        self.patch_width =  patch_width
        self.target_vocab_size = target_vocab_size

        # Layers
        self.patchify = CreatePatches(self.patch_width)
        self.patch_encoder = PatchEncoder()

        self.pos_embedding = RelativePositionEmbedding(176)
        self.sequence_encoder = TransformerEncoder(8, 8, 176, 4)
        self.sequence_decoder = TransformerDecoder(8, 8, 176, 128, self.target_vocab_size)
        self.classifier = layers.Dense(target_vocab_size)

        # Loss
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    def pad_mask(self, src, dest):
        batch_size = src.shape[0]
        n_src = src.shape[1]
        n_dest = dest.shape[1]
        mask = tf.ones((batch_size, n_dest, n_src), dtype=tf.bool)

        return mask

    def embed_speech(self, specs):
        # Add channels dimension
        images = tf.expand_dims(specs, -1)

        # Split into patches
        patches = self.patchify(images)

        # Create features from sequences of patches
        features = self.patch_encoder(patches)

        return features


    def call(self, inputs, training):
        # Inputs consists of spectrograms, and the target sentence as integer encoded tokens
        inp, tar = inputs

        features = self.embed_speech(inp)

        # Masks
        enc_padding_mask = self.pad_mask(features, features)
        dec_padding_mask = self.pad_mask(features, tar)

        encoded = self.sequence_encoder(features, training, enc_padding_mask)
        decoded, attention_weights = self.sequence_decoder(tar, encoded, training, dec_padding_mask)

        pred = self.classifier(decoded)

        return pred

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source, target = batch

        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        # Dec_input: "BOF ... ... ..."
        # Dec_target: "... ... ... EOF"

        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.target_vocab_size)

            # Mask out padded in loss
            mask = tf.math.logical_not(tf.math.equal(dec_target, 1))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result()}

    def translate(self, source, steps, bos_idx):
        """Perform ASR on batched spectrograms by greedily
        predicting one token at a time using previous predictions."""
        batch_size = source.shape[0]

        x = self.embed_speech(source)

        dec_inputs = tf.ones((batch_size, 1), dtype=tf.int32) * bos_idx

        enc_padding_mask = self.pad_mask(x, x)
        dec_padding_mask = self.pad_mask(x, dec_inputs)

        encoded = self.sequence_encoder(x, False, enc_padding_mask)
        dec_logits = []
        for i in range(steps):
            decoded, _ = self.sequence_decoder(dec_inputs, encoded, False, dec_padding_mask)
            logits = self.classifier(decoded)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_inputs = tf.concat([dec_inputs, last_logit], axis=-1)

        return dec_inputs

class TransformerEncoder(layers.Layer):
    def __init__(self, n_blocks, n_heads, d_model, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.pos_embedding = RelativePositionEmbedding(self.d_model)
        self.enc_blocks = [TransformerEncoderBlock(self.d_model, self.n_heads, self.ff_dim, self.dropout_rate)
                           for _ in range(self.n_blocks)]

        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, emb_x, training, padding_mask):
        seq_len = emb_x.shape[1]
        pos_emb = self.pos_embedding(emb_x)
        x = emb_x + pos_emb
        x = self.dropout(x, training=training)

        for i, block in enumerate(self.enc_blocks):
            x = block(x, training, padding_mask)

        return x

class TransformerDecoder(layers.Layer):
    def __init__(self, n_blocks, n_heads, d_model, ff_dim, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.embedding = layers.Embedding(target_vocab_size, self.d_model)
        self.pos_embedding = RelativePositionEmbedding(self.d_model)

        self.dec_blocks = [TransformerDecoderBlock(self.d_model, self.n_heads, self.ff_dim)
                           for _ in range(self.n_blocks)]

        self.dropout = layers.Dropout(self.dropout_rate)

    def lookahead_mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat([tf.expand_dims(batch_size, -1),
                          tf.constant([1,1], dtype=tf.int32)], 0)
        out = tf.tile(mask, mult)

        return out

    def call(self, tar, enc_output, training, padding_mask):
        batch_size = enc_output.shape[0]
        src_seq_len = enc_output.shape[1]
        dest_seq_len = tar.shape[1]
        
        lookahead_mask = self.lookahead_mask(batch_size, dest_seq_len, dest_seq_len, tf.bool)

        attention_weights = {}

        # Embed targets
        emb_tar = self.embedding(tar)
        pos_tar = self.pos_embedding(tar)
        emb_tar = emb_tar + pos_tar

        x = self.dropout(emb_tar, training=training)

        for i, block in enumerate(self.dec_blocks):
            x, block1, block2 = block(x, enc_output, lookahead_mask, padding_mask, training)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights

class TransformerEncoderBlock(layers.Layer):
    """Encoder block with masked multihead attention and the normalization, dropout layers."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        """Simple feed forward network."""
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def build(self, input_shape):
        self.attn_multi = layers.MultiHeadAttention(self.n_heads, self.d_model, dropout=self.dropout_rate)
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.attn_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ffn = self.point_wise_feed_forward_network(self.d_model, self.ff_dim)
        self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, x, training, mask): 
        attn_layer = self.attn_multi(x,x,x, mask, training=training)
        attn_layer = self.attn_dropout(attn_layer, training=training)
        out1 = self.attn_normalize(x + attn_layer)

        ff_layer = self.ffn(out1)
        ff_layer = self.ff_dropout(ff_layer, training=training)
        out2 = self.ff_normalize(out1 + ff_layer)
        
        return out2

class TransformerDecoderBlock(layers.Layer):
    """Decoder block that uses masked attention (look-ahead masks and padding mask)."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        """Simple feed forward network."""
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


    def build(self, input_shape):
        self.attn_multi1 = layers.MultiHeadAttention(self.n_heads, self.d_model, dropout=self.dropout_rate)
        self.attn_multi2 = layers.MultiHeadAttention(self.n_heads, self.d_model, dropout=self.dropout_rate)

        self.ffn = self.point_wise_feed_forward_network(self.d_model, self.ff_dim)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, target, enc_output, lookahead_mask, padding_mask, training):
        attn1, attn_scores1 = self.attn_multi1(target, target, target,
                                               lookahead_mask, return_attention_scores=True, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.normalize1(attn1 + target)

        attn2, attn_scores2 = self.attn_multi2(out1, enc_output, enc_output,
                                               padding_mask, return_attention_scores=True, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.normalize2(attn2 + out1)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.normalize3(ffn_out + out2)

        return out3, attn_scores1, attn_scores2


class CreatePatches(layers.Layer):
    """Split spectrogram into same width patches. No overlap currently"""
    def __init__(self, patch_width):
        super().__init__()
        self.patch_width = patch_width

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patch_height = images.shape[2]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1,self.patch_width, patch_height, 1],
            strides=[1,self.patch_width, patch_height, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = tf.reshape(patches, [batch_size,
                                       -1,
                                       self.patch_width,
                                       patch_height,
                                       1,
                                       ])

        # Shape (batch, seq, height, width)
        return patches

class PatchEncoder(layers.Layer):
    """Given a sequence of patches, encode each patch individually with convolution stack"""
    def __init__(self):
        super().__init__()
        self.patch_conv1 = tf.keras.layers.Conv2D(
            2, (5,3), (2,2), activation='relu')
        self.patch_conv2 = tf.keras.layers.Conv2D(
            4, (5,3), (2,2), activation='relu')
        self.patch_conv3 = tf.keras.layers.Conv2D(
            16, (5,3), (2,2), activation='relu')

    def call(self, patches):
        """Encode patches of shape (batch, seq, height, width, channel)"""
        # Swap batch and seq dims
        patches = tf.transpose(patches, [1,0,2,3,4])
        # Run each conv on each sequence elemnent
        out = tf.map_fn(self.patch_conv1, patches)
        out = tf.map_fn(self.patch_conv2, out)
        out = tf.map_fn(self.patch_conv3, out)
        # Flatten into feature vectors
        features = tf.transpose(out, [1,0,2,3,4])
        features = tf.reshape(features, (features.shape[0], features.shape[1], -1))

        return features

class RelativePositionEmbedding(layers.Layer):
    """Provides the transformer with sequence
    order information, which is important because without recurrent
    networks, this information is lost."""
    def __init__(self, hidden_size, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        super().__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def call(self, inputs, length=None):
        input_shape = inputs.shape
        length = input_shape[1]

        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = (tf.math.log(float(max_timescale) / float(min_timescale)) /
                                   (tf.cast(num_timescales, tf.float32) - 1))
        
        inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) *
                                                -log_timescale_increment)
        
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        
        position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

        return position_embeddings
