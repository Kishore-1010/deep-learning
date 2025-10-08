Code:

import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from keras.utils import to_categorical
sentences = [['I', 'love', 'NLP'], ['He', 'plays', 'football']]
tags = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]
words = sorted(list(set(word for sent in sentences for word in sent)))
tags_vocab = sorted(list(set(tag for tag_list in tags for tag in tag_list)))

word2idx = {w:i+1 for i, w in enumerate(words)}  
tag2idx = {t:i for i, t in enumerate(tags_vocab)}
idx2tag = {i:t for t,i in tag2idx.items()}

max_len = max(len(s) for s in sentences)
n_words = len(word2idx) + 1
n_tags = len(tag2idx)

# Encode sentences
X = np.array([[word2idx[w] for w in s] + [0]*(max_len-len(s)) for s in sentences])
y = np.array([[tag2idx[t] for t in tag_list] + [0]*(max_len-len(tag_list)) for tag_list in tags])
y = to_categorical(y, num_classes=n_tags)

# -----------------------------
# 2️⃣ Build Model
# -----------------------------
input_layer = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=64, input_length=max_len)(input_layer)
model = Bidirectional(LSTM(units=64, return_sequences=True))(model)
output = TimeDistributed(Dense(n_tags, activation='softmax'))(model)

model = Model(input_layer, output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 3️⃣ Train
# -----------------------------
model.fit(X, y, batch_size=2, epochs=300, verbose=0)

# -----------------------------
# 4️⃣ Predict
# -----------------------------
def predict_pos(sent):
    seq = np.array([[word2idx[w] for w in sent]])
    preds = model.predict(seq)
    tags_pred = [idx2tag[np.argmax(p)] for p in preds[0][:len(sent)]]
    return tags_pred

# -----------------------------
# 5️⃣ Test
# -----------------------------
test_sentence = ['I', 'love', 'NLP']
print("Input:", test_sentence)
print("Predicted POS tags:", predict_pos(test_sentence))

Output :

<img width="1599" height="899" alt="Screenshot 2025-10-08 120129" src="https://github.com/user-attachments/assets/9528ef51-7aed-435e-8f10-430bf14544a5" />
<img width="1599" height="899" alt="Screenshot 2025-10-08 120136" src="https://github.com/user-attachments/assets/3944551c-898a-471c-b085-f70e33f47020" />
