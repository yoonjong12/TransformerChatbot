from keras import backend
from tensorflow import keras
import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer


# 기존 keraas 모델을 로드합니다.
def modelLoad(path, MAX_LENGTH):
    # tf 환경이 꼬이는 것을 방지합니다.
    tf.keras.backend.clear_session()
    # 각 item들은 custom한 keras layer로써 모델 호출 시 모두 선언해줘야합니다.
    custom_objects = {'PositionalEncoding': PositionalEncoding,
                      'MultiHeadAttention': MultiHeadAttention,
                      'CustomSchedule': CustomSchedule,
                      'loss_function': customLoss(MAX_LENGTH),
                      'accuracy': custom_accuracy(MAX_LENGTH),
                      'create_padding_mask': create_padding_mask,
                      'backend': backend,
                      'tf': tf}
    # custom_objects에 인자로 dictionary를 넣어줍니다
    return keras.models.load_model(path, custom_objects=custom_objects)


#### 전처리 및 모델 코드 ####
class Preprocess:
    def __init__(self):
        super().__init__()

    # 토크나이저를 만듭니다.
    # 토크나이저는 말뭉치로부터 단어 사전을 만들고 번호를 부여합니다.
    def buildTokenizer(self, questions, answers):
        self.questions = questions
        self.answers = answers
        # 먼저 질문과 답변을 모두 합쳐서 말뭉치를 만듭니다.
        corpus = questions + answers
        # /를 살려서 만든 이유는 나중에 Khaiii를 사용할 때 이 문자가 필요하기 때문입니다. Khaiii를 사용안하거나 삭제하고 싶으면 추가하세요
        tk = Tokenizer(filters='!"#$%&()*+,-.s:;<=>?@[\\]^_`{|}~\t\n', lower=False)  # / 만 살렸다, 소문자변환 안한다.
        tk.fit_on_texts(corpus)
        VOCAB_SIZE = len(tk.word_index)
        # 토크나이저의 맨 뒤에 각각 SOS, EOS토큰을 추가해줍니다.
        self.START_TOKEN, self.END_TOKEN = [VOCAB_SIZE], [VOCAB_SIZE + 1]
        # 앞에서 토큰 2개가 추가되었으니 사전의 크기도 2만큼 커졌다고 말해줍니다.
        self.VOCAB_SIZE = VOCAB_SIZE + 2
        return tk

    # build한 토크나이저를 저장합니다.
    def saveTokenizer(self, directory, tokenizer):
        tokenizer.to_json()
        with open(directory, 'w', encoding='UTF-8-sig') as f:
            f.write(json.dumps(vars(tokenizer), ensure_ascii=False))

    # 토크나이저를 로드합니다.
    def loadTokenzier(self, directory):
        with open(directory, encoding='UTF-8-sig') as fh:
            data = json.load(fh)
        tk = Tokenizer()
        key = list(data.keys())
        for i in key:
            setattr(tk, i, data[i])
        VOCAB_SIZE = len(Tokenizer(tk).word_index) + 1
        self.START_TOKEN, self.END_TOKEN = [VOCAB_SIZE], [VOCAB_SIZE + 1]
        self.VOCAB_SIZE = VOCAB_SIZE + 2
        # 토크나이저 로드하면 모든 key,value가 string으로 들어감 나중에 토큰을 텍스트로
        # 복원할 때 정상적으로 구동하기 위해서 index_word는 key를 int로 바꿔줌
        tk.index_word = {int(k): v for k, v in tk.index_word.items()}
        return tk

    # 리스트 형태를 토큰 모델에 넣어주려고 text로 만듦
    def list2text(self, x):
        return [e for s in x for e in s]

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, questions, answers, tokenizer, MAX_LENGTH=30):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(questions, answers):
            # tokenize sentence
            sentence1 = [self.START_TOKEN] + tokenizer.texts_to_sequences(sentence1) + [self.END_TOKEN]
            sentence2 = [self.START_TOKEN] + tokenizer.texts_to_sequences(sentence2) + [self.END_TOKEN]
            # check tokenized sentence max length
            if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
                tokenized_inputs.append(self.list2text(sentence1))
                tokenized_outputs.append(self.list2text(sentence2))

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
                            tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
                            tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
        return tokenized_inputs, tokenized_outputs

    # 학습하기 위한 Dataset을 준비합니다.
    def buildDataset(self, inputs, outputs, BATCH_SIZE, BUFFER_SIZE=20000):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': inputs,
                'dec_inputs': inputs[:, :-1]
            },
            {
                'outputs': outputs[:, 1:]
            },
        ))
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


# 이 밑은
# Tensorflow2.0을 기반으로 한 Transformer 모듈입니다.
# Keras의 layer를 class로 직접 구현하면 필수로 포함시켜야 하는 요소들이 있습니다.
# get_config나 call 등이 그러합니다.
############################
#### MultiHeadAttention ####
############################

############################
#### MultiHeadAttention ####
############################
def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth) # scaling 

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1) 

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention", **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = self.d_model // self.num_heads
        assert d_model % self.num_heads == 0
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
                        "num_heads":self.num_heads, 
                        "d_model":self.d_model,
                       })
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[ 
                                    'value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)
        return outputs

#################
#### Masking ####
#################
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

######################
#### Pos Encoding ####
######################
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, name='PositionalEncoding', **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.position = position
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(tf.constant(10000,dtype='float32'), (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
                            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config().copy()
        config.update({
                        'd_model': self.d_model,
                        'position': self.position
                        })
        return config

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

###########################
#### 인코더 내부 Layer ####
###########################
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)
  
##############################################
#### 임베딩 + Pos Encoding + 인코더 Layer ####
##############################################
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

###########################
#### 디코더 내부 Layer ####
###########################
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(inputs={
                    'query': inputs,
                    'key': inputs,
                    'value': inputs,
                    'mask': look_ahead_mask
                })
    attention1 = tf.keras.layers.LayerNormalization(
    epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(inputs={
                    'query': attention1,
                    'key': enc_outputs,
                    'value': enc_outputs,
                    'mask': padding_mask
                })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
    epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
    epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
                    inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                    outputs=outputs,
                    name=name)

###################################################
#### 임베딩 + Pos Encoding + 디코더 내부 Layer ####
###################################################
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
                    units=units,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    name='decoder_layer_{}'.format(i),
                )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
                        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                        outputs=outputs,
                        name=name)

#######################################
#### 인코더 + 디코더 = Transformer ####
#######################################
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    units=units,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    units=units,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
    
#######################################
######### Loss & LearningRate #########
#######################################
def customLoss(MAX_LENGTH, name='customLoss'):
    def loss_function(y_true, y_pred, name='loss_function'):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)
    return loss_function
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, name='CustomSchedule', **kwargs):
        super(CustomSchedule, self).__init__(**kwargs)

        self.d_m = d_model
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
                  'd_model': self.d_m,
                  'warmup_steps': self.warmup_steps,
                 }
        return config 

def custom_accuracy(MAX_LENGTH, name='custom_accuracy'):
    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    return accuracy

#### 예측 ####
def transform(sentence, analyzer, anl_type):
    if anl_type == 'Konlpy_morphs':
        return [[i] for i in  analyzer.morphs(sentence)]
    elif anl_type == 'Konlpy_pos':
        return [[i] for i in  analyzer.pos(sentence, join=True)]
    elif anl_type == 'Konlpy_tokens':
        return [[i] for i in sentence]
    elif anl_type == 'Khaiii':
        sentence = [str(i).split('\t')[1] for i in  analyzer.analyze(sentence)]
        return [e for s in sentence for e in s.split(' + ')]
    else:
        print('use Konlpy or Khaiii')

def evaluate(model, tokenizer, sentence, analyzer, anl_type, MAX_LENGTH=30):
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    START_TOKEN, END_TOKEN = [VOCAB_SIZE], [VOCAB_SIZE + 1]
    VOCAB_SIZE += 2

    sentence = [START_TOKEN] + tokenizer.texts_to_sequences(transform(sentence, analyzer, anl_type)) + [END_TOKEN]
    sentence = tf.expand_dims([e for s in sentence for e in s],axis=0)
    output = tf.expand_dims(START_TOKEN, 0)
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        output = tf.concat([output, predicted_id], axis=-1)

    output = tf.squeeze(output, axis=0)
    return tokenizer.sequences_to_texts(
                        [[i] for i in output.numpy()])

def pos2text(token_list):
    pure_input = [i[:i.index('/')] for i in token_list]
    # 띄어쓰기 교정
    # Josa : 조사, PreEomi : 선어말어미, Eomi : 어미, Suffix : 접미사
    pos_list = ['Josa','PreEomi','Eomi','Suffix']
    for i in range(len(token_list)):
        p = token_list[i]
        if p[p.index('/')+1:] in pos_list:
            # '/토큰'으로 만들어서 나중에 앞 단어와 붙일 거임
            pure_input[i] = '/'+pure_input[i]
    pure_text = ' '.join(pure_input)
    pure_text = pure_text.replace(' /','')

    # Rule 추가
    pure_text = pure_text.replace('중 증','중증')
    pure_text = pure_text.replace(' 은','은')
    pure_text = pure_text.replace(' 로','로')
    pure_text = pure_text.replace(' 와','와')


    return pure_text

def token2text(token_list):
    return ' '.join(token_list)
def text2token(x, analyzer):
    return analyzer.pos(x, join=True)
