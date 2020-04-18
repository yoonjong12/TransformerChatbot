# TransformerChatbot
트랜스포머(Transformer)를 사용한 한국어 챗봇

필요환경 : Tensorflow2.0, Konlpy <br>
참조 : https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2

Tensorflow 블로그의 Transformer Chatbot Tutorial 게시글을 클래스로 활용할 수 있도록 수정하였습니다.


## 시작하기

패키지의 main.py를 실행하면 즉시 조잡하게나마 테스트할 수 있습니다.
> Q : (입력하고 싶은 질문) <br>
> A : 트랜스포머의 답변 <br>
> (종료는 'quit' 입력) <br>

학습 말뭉치가 특정 주제에 관한 데이터이기 때문에 일상대화가 불가능 할 수 있습니다. <br>
학습 말뭉치 예시 일부를 'data' 디렉토리에 csv로 올려놓았으니 참고하셔서 테스트 해보시기 바랍니다.

## 학습시키기
직접 말뭉치를 준비하고, 모델을 학습시키는 과정을 구글 Colab 노트북으로 정리했습니니다.
https://github.com/yoonjong12/TransformerChatbot/blob/master/Train_Transformer.ipynb
