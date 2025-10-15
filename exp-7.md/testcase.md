import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

# =======================
# CONFIGURATION
# =======================
vocab_inp_size = 5000    # English vocab size
vocab_tar_size = 5000    # Hindi vocab size
embedding_dim = 256
units = 512
BATCH_SIZE = 64
MAX_LENGTH = 15

# =======================
# ENCODER
# =======================
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.enc_units, return_sequences=True, ret<img width="485" height="558" alt="Screenshot 2025-10-15 101641" src="https://github.com/user-attachments/assets/708af185-6a84-49c4-b8d8-fc630c749f05" />
<img width="422" height="592" alt="Screenshot 2025-10-15 101626" src="https://github.com/user-attachments/assets/d7480ffd-1667-4433-8ded-b4959b5cc09d" />
urn_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c

# =======================
# BAHADANAU ATTENTION
# =======================
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# =======================
# DECODER
# =======================
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state_h, state_c, attention_weights

# =======================
# INSTANTIATE MODELS
# =======================
encoder = Encoder(vocab_inp_size, embedding_dim, units)
decoder = Decoder(vocab_tar_size, embedding_dim, units)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# =======================
# LOSS FUNCTION
# =======================
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# =======================
# TRAIN STEP (Teacher Forcing)
# =======================
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden_h, enc_hidden_c = encoder(inp)
        dec_hidden_h, dec_hidden_c = enc_hidden_h, enc_hidden_c
        dec_input = tf.expand_dims([1] * BATCH_SIZE, 1)  # <start> token

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden_h, dec_hidden_c, _ = decoder(dec_input, dec_hidden_h, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

# =======================
# TRAINING (Demo with random data)
# =======================
EPOCHS = 3
sample_inp = np.random.randint(0, vocab_inp_size, (BATCH_SIZE, MAX_LENGTH))
sample_targ = np.random.randint(0, vocab_tar_size, (BATCH_SIZE, MAX_LENGTH))

for epoch in range(EPOCHS):
    enc_hidden = [tf.zeros((BATCH_SIZE, units)), tf.zeros((BATCH_SIZE, units))]
    batch_loss = train_step(sample_inp, sample_targ, enc_hidden)
    print(f"Epoch {epoch+1} Loss {batch_loss.numpy():.4f}")

# =======================
# INFERENCE / TRANSLATION
# =======================
def translate(sentence_tokens):
    inputs = tf.convert_to_tensor(sentence_tokens)
    enc_output, enc_hidden_h, enc_hidden_c = encoder(inputs)
    dec_hidden_h, dec_hidden_c = enc_hidden_h, enc_hidden_c
    dec_input = tf.expand_dims([1], 0)  # <start> token
    result = []

    for t in range(MAX_LENGTH):
        preds, dec_hidden_h, dec_hidden_c, attention_weights = decoder(dec_input, dec_hidden_h, enc_output)
        predicted_id = tf.argmax(preds[0]).numpy()
        result.append(predicted_id)
        if predicted_id == 2:  # <end> token
            break
        dec_input = tf.expand_dims([predicted_id], 0)
    return result

# Simulate translation
dummy_sentence = np.random.randint(0, vocab_inp_size, (1, MAX_LENGTH))
predicted_tokens = translate(dummy_sentence)
idx2word = {i: f"w{i}" for i in range(vocab_tar_size)}
translated_sentence = ' '.join([idx2word[i] for i in predicted_tokens])
print("Predicted Hindi:", translated_sentence)

# =======================
# EVALUATION TABLE (Demo)
# =======================
input_sentences = ["How are you?", "I love coding."]
predicted_outputs = ["तुम कैसे हो?", "मुझे कोडिंग पसंद है।"]
expected_outputs = ["तुम कैसे हो?", "मुझे कोडिंग पसंद है।"]

results = []
for inp, pred, exp in zip(input_sentences, predicted_outputs, expected_outputs):
    correct = "Y" if pred.strip() == exp.strip() else "N"
    results.append([inp, pred, correct])

df = pd.DataFrame(results, columns=["Input Sentence (English)", "Predicted Output (Hindi)", "Correct (Y/N)"])
print("\nEvaluation Table:\n")
print(df.to_string(index=False))
df.to_csv("translation_results.csv", index=False, encoding="utf-8-sig")
print("\n✅ Results saved to 'translation_results.csv'")<img width="485" height="558" alt="Screenshot 2025-10-15 101641" src="https://github.com/user-attachments/assets/0dadd07e-f2c6-4b69-b591-f81fc5cbe447" />
<img width="422" height="592" alt="Screenshot 2025-10-15 101626" src="https://github.com/user-attachments/assets/fbe074cc-1da3-4bc1-b6aa-e8c59d5866a3" />
