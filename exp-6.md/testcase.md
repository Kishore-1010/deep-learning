Code :

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = ["i love nlp", "he plays football"]
tags = [["pron", "verb", "noun"], ["pron", "verb", "noun"]]
word2idx = {"<pad>": 0, "<unk>": 1}
for sentence in sentences:
  for word in sentence.split():
    if word not in word2idx:
      word2idx[word] = len(word2idx)

tag2idx = {"<pad>": 0, "<unk>": 1}
for tag_sequence in tags:
  for tag in tag_sequence:
    if tag not in tag2idx:
      tag2idx[tag] = len(tag2idx)

X = [[word2idx.get(word, word2idx["<unk>"]) for word in sentence.split()] for sentence in sentences]
Y = [[tag2idx.get(tag, tag2idx["<unk>"]) for tag in tag_sequence] for tag_sequence in tags]

max_len = max(len(seq) for seq in X)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_padded = pad_sequences(X, maxlen=max_len, padding='pre', value=word2idx["<pad>"])
Y_padded = pad_sequences(Y, maxlen=max_len, padding='pre', value=tag2idx["<pad>"])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

vocab_size = len(word2idx)
tag_size = len(tag2idx)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dense(units=tag_size, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_padded, Y_padded, epochs=10, batch_size=32)
new_sentences = ["i love football", "he loves nlp"]
# Assuming corresponding actual tags for the new sentences (replace with your actual data if available)
# For demonstration purposes, let's create some dummy actual tags
actual_tags_for_new_sentences = [["pron", "verb", "noun"], ["pron", "verb", "noun"]]


# Tokenize new sentences
X_new = [[word2idx.get(word, word2idx["<unk>"]) for word in sentence.split()] for sentence in new_sentences]

# Pad new sentences
X_new_padded = pad_sequences(X_new, maxlen=max_len, padding='pre', value=word2idx["<pad>"])

# Predict tags
predictions = model.predict(X_new_padded)

# Convert predictions to tag indices
predicted_tag_indices = np.argmax(predictions, axis=-1)

# Convert tag indices to tag strings
idx2tag = {v: k for k, v in tag2idx.items()}
predicted_tags = [[idx2tag[idx] for idx in seq] for seq in predicted_tag_indices]

# Prepare data for table display
results = []
for i, sentence in enumerate(new_sentences):
    original_sentence = sentence
    predicted_tag_sequence = predicted_tags[i]
    # Get actual tags for comparison
    actual_tag_sequence = actual_tags_for_new_sentences[i]

    # Determine if prediction is correct (simple exact match for now, can be improved)
    is_correct = "Y" if predicted_tag_sequence == actual_tag_sequence else "N"


    results.append({
        "Original Sentence": original_sentence,
        "Predicted Tags": " ".join(predicted_tag_sequence),
        "Correct (Y/N)": is_correct
    })

# Display results in a table
df_results = pd.DataFrame(results)
display(df_results)

Output :

<img width="1599" height="896" alt="Screenshot 2025-10-08 120049" src="https://github.com/user-attachments/assets/3685a91d-2474-4eaf-ae73-22c0d11ec058" />
<img width="1599" height="899" alt="Screenshot 2025-10-08 120104" src="https://github.com/user-attachments/assets/1c991732-3201-43db-b47b-c5391587248e" />
