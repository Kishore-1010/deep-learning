import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Sample Shakespeare-like data (use a real corpus for better results)
data = [
    "To be or not to be that is the question",
    "Whether tis nobler in the mind to suffer",
    "The slings and arrows of outrageous fortune",
    "Or to take arms against a sea of troubles",
    "And by opposing end them"
]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

# Create predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode labels
y = to_categorical(y, num_classes=total_words)

# Build the LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_seq_len - 1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for 50 epochs
model.fit(X, y, epochs=50, verbose=1)
<img width="407" height="656" alt="Screenshot 2025-08-20 102219" src="https://github.com/user-attachments/assets/ba8928e6-a168-4693-a986-edf85414a83f" />
