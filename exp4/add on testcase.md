import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Sample Shakespeare-like data (expand with more lines for better accuracy)
data = [
    "To be or not to be",
    "What light through yonder window breaks",
    "It is the east and Juliet is the sun",
    "Arise fair sun and kill the envious moon",
]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to the same length
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

# Split predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode the labels
y = to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Function to predict the next word
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted_idx = model.predict(token_list, verbose=0).argmax(axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_idx:
            return word
    return None

# Test sequences and expected next words
test_data = [
    ("To be or not", "to"),
    ("What light through yonder window", "breaks")
]

print(f"{'Input Sequence':<35}{'Predicted Word':<20}{'Correct (Y/N)'}")
for seq, expected_word in test_data:
    predicted_word = predict_next_word(seq.lower())
    correct = 'Y' if predicted_word == expected_word else 'N'
    print(f"{seq:<35}{predicted_word:<20}{correct}")
<img width="592" height="60" alt="Screenshot 2025-08-20 102420" src="https://github.com/user-attachments/assets/14405fc4-360f-423e-be41-46cb43e9eafc" />
