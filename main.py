from module import Transformer
from config import *
from konlpy.tag import Okt
import tensorflow as tf

def startVerOne():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.enable_eager_execution(config=config)
if tf.__version__[0] == '1':
    print('Using TF ver.1')
    startVerOne()
else:
    print('Using TF 2.0')


t = Okt()

weights = os.path.join(weights_dir, weights_name)
tokenizer = os.path.join(vocab_dir, vocab_name)

model = Transformer.modelLoad(weights, MAX_LENGTH=30)
tokenizer = Transformer.tokenizerLoad(tokenizer)



# 예측

while True:
    sentence = input('Q : ')
    if sentence == 'quit':
        break
    else:
        prediction = Transformer.evaluate(model, tokenizer, sentence, t, anl_type='Konlpy_tokens', MAX_LENGTH=30)
        new_text = ' '.join(prediction)

        print(f'A : {new_text}')# 예측z