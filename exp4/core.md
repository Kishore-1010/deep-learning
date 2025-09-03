import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# Sample text data
data = [
    "Deep learning is amazing",
    "Deep learning builds intelligent systems"
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

# Create predictors and label
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode labels
y = to_categorical(y, num_classes=total_words)

# Build the model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(SimpleRNN(50))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0).argmax(axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return None

# Testing predictions like in the image
texts = [
    "Deep",
    "Deep learning is",
    "Deep learning is amazing.",
    "Deep learning is amazing. Deep",
    "Deep learning is amazing. Deep learning",
    "Deep learning is amazing. Deep learning builds",
    "Deep learning is amazing. Deep learning builds intelligent"
]

for txt in texts:
    next_word = predict_next_word(model, tokenizer, txt.lower(), max_seq_len)
    print(f"['{txt}'] -> '{next_word}'")
<img width="956" height="134" alt="Screenshot 2025-08-20 102154" src="https://github.com/user-attachments/assets/4eef9cb6-07c9-4e2f-9f75-77c7090f65f1" />
