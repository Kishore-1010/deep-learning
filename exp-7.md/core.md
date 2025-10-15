from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

# Example values (you must define these before)
num_encoder_tokens = 10000  # vocab size for encoder
num_decoder_tokens = 10000  # vocab size for decoder

# Define encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Define decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(num_decoder_tokens, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
<img width="665" height="682" alt="Screenshot 2025-10-15 101259" src="https://github.com/user-attachments/assets/2a5903b1-a8b0-4b7c-b238-fe648be542c6" />
