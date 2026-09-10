import keras_nlp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MAX_SEQUENCE_LENGTH = 50
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512
NUM_HEADS = 2

input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)

x = keras.layers.GlobalAveragePooling1D()(x)

model = keras.Model(inputs=input_ids, outputs=x)
