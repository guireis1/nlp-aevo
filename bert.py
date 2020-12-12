import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tokenization

# Função para embendding e criação dos futuros inputs do modelo Bert
def bert_encode(texts, tokenizer, max_len):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"] # é necessário a adição do [CLS] no começo do texto e [SEP] no final do texto
        pad_len = max_len - len(input_sequence) # o pad é para padronizar os vetores, de acordo com o len definido 
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) # convertendo para ID
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)  # Token ID
        all_masks.append(pad_masks)   # Mask para chamar atenção no que é palavra e no que é pad! Ajuda o treinamento do modelo!
        all_segments.append(segment_ids) # Uma das formas de aprendizagem do modelo Bert é dividindo o texto em dois seguimentos para futuras comparações! Esse input mostra quem pertence ao primeiro segmento e quem pertençe ao segundo!
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    
    if Dropout_num == 0:   #optei por sem dropout!
        out = Dense(1, activation='sigmoid')(clf_output)
    else:
        x = Dropout(Dropout_num)(clf_output)
        out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

#Baixando o modelo do TFHub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

train = pd.read_csv("data/train_desastre.csv")
train = pd.read_csv("data/test_desastre.csv")

# tokenizando todo o texto, usando o vocab do bert baixado no tfhub e colocando em lower_case.
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)   

train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values

model_BERT = build_model(bert_layer, max_len=160)
model_BERT.summary()

#Treino
checkpoint = ModelCheckpoint('data/model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16
)

model.load_weights('data/model.h5')
test_pred = model.predict(test_input)

submission = pd.DataFrame(test_pred.round().astype(int),columns=['target'])
submission.to_csv('data/test_pred.csv', index=False)