Code :

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
input_texts=['I love NLP','He plays football']
target_texts=[['PRON','VERB','NOUN'],['PRON','VERB','NOUN']]
word_vocab=sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab=sorted(set(tag for tags in target_texts for tag in tags))
word2idx={word:i+1 for i,word in enumerate(word_vocab)}
tag2idx={tag:i for i,tag in enumerate(tag_vocab)}
idx2tag={i:tag for tag,i in tag2idx.items()}
encoder_input_data=np.array([[word2idx[word] for word in sent.split()] for sent in input_texts])
decoder_output_data=np.array([[tag2idx[tag] for tag in tags] for tags in target_texts])
decoder_input_data=np.zeros_like(decoder_output_data)
vocab_size=len(word2idx)+1
tag_size=len(tag2idx)
encoder_inputs=Input(shape=(None,))
encoder_emb_layer=Embedding(input_dim=vocab_size,output_dim=64)
encoder_embedding=encoder_emb_layer(encoder_inputs)
encoder_lstm=LSTM(64,return_state=True)
_,state_h,state_c=encoder_lstm(encoder_embedding)
encoder_states=[state_h,state_c]
decoder_inputs=Input(shape=(None,))
decoder_emb_layer=Embedding(input_dim=tag_size+1,output_dim=64)
decoder_embedding=decoder_emb_layer(decoder_inputs)
decoder_lstm=LSTM(64,return_sequences=True,return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_embedding,initial_state=encoder_states)
decoder_dense=Dense(tag_size,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)
model=Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit([encoder_input_data,decoder_input_data],decoder_output_data,epochs=500,verbose=0)
encoder_model=Model(encoder_inputs,encoder_states)
decoder_state_input_h=Input(shape=(64,))
decoder_state_input_c=Input(shape=(64,))
decoder_states_inputs=[decoder_state_input_h,decoder_state_input_c]
dec_emb2=decoder_emb_layer(decoder_inputs)
decoder_outputs2,state_h2,state_c2=decoder_lstm(dec_emb2,initial_state=decoder_states_inputs)
decoder_states2=[state_h2,state_c2]
decoder_outputs2=decoder_dense(decoder_outputs2)
decoder_model=Model([decoder_inputs]+decoder_states_inputs,[decoder_outputs2]+decoder_states2)
def predict_pos_tags(sentence):
    seq=np.array([[word2idx[word] for word in sentence.split()]])
    states_value=encoder_model.predict(seq)
    target_seq=np.zeros((1,len(sentence.split())),dtype='int32')
    decoded_tags=[]
    for i in range(len(sentence.split())):
        output_tokens,h,c=decoder_model.predict([target_seq]+states_value)
        sampled_token_index=np.argmax(output_tokens[0,i,:])
        decoded_tags.append(idx2tag[sampled_token_index])
        states_value=[h,c]
    return decoded_tags
print("Input: I love NLP")
print("Predicted POS tags:",predict_pos_tags("I love NLP"))

Output :

<img width="1599" height="895" alt="Screenshot 2025-10-08 120113" src="https://github.com/user-attachments/assets/37300334-23a1-469a-bcb0-2f8892519513" />
<img width="1599" height="899" alt="Screenshot 2025-10-08 120121" src="https://github.com/user-attachments/assets/758c3dd8-1e48-4b10-8e1f-b667806a3990" />
