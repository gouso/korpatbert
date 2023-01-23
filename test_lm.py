#-*- coding: utf-8 -*-
# 프로그램 목적 : KorPatBERT 와 Keras를 활용하여, 3개의 클래스로 문장을 분류하는 테스트 프로그램
# 사용되어진 데이터 셋은 label, sentence 두개의 컬럼으로 구성되어 있으며,
# 클래스별 300개씩 총 900개의 데이터로 이루어져 있다.
# label은 특허 CPC코드 섹션 "A", "B", "C" 3개의 클래스로 구성되어 있고, 
# sentence 는 특허기술에 대한 문장으로 이루어져 있다.

# 필요 라이브러리 임포트
import tensorflow as tf
import bert
import os
import mecab
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tqdm import tqdm
from korpat_tokenizer import Tokenizer

# 필요 환경변수 설정
os.environ['TF_KERAS'] = '1'    # Keras Tensorflow 설정
config_path     = "./pretrained/korpat_bert_config.json" # KorpatBert Config 파일 경로
vocab_path      = "./pretrained/korpat_vocab.txt"        # KorpatTokenizer Vocabulary 파일 경로
checkpoint_path = "./pretrained/model.ckpt-381250"       # KorpatBert 모델파일 경로
pretreind_model_dir = "./pretrained/"                    # KorpatBert 모델 디렉토리 경로
dataset_path = "./lm_test_data.tsv"                      # 사용할 데이터셋 경로
save_model_path = "./korpat_bert_test_model.h5"          # 학습완료 모델 저장 경로

MAX_SEQ_LEN = 512 # 학습 최대 토큰 갯수
BATCH_SIZE = 4    # 학습 배치 사이즈 기본8
EPOCHS = 3        # 학습 에폭 기본 1
LR = 0.00003      # 학습률

# 분리 비율에 따른 데이터셋 분리 함수
# 입력 : dataset(분리 대상 데이터셋), split_val(분리 비율)
# 출력 : train_data(분리 데이터셋), dev_data(분리 데이터셋)
def dataset_split(dataset, split_val):
    lengths = int(len(dataset) * split_val)
    train_data = dataset[:lengths]
    dev_data = dataset[lengths:]
    return train_data, dev_data

# 데이터 셋 로드 함수
# 입력 : dataset_url (전체 데이터셋 경로)
# 출력 : train_data(학습데이터) dev_data(검증데이터), test_data(평가데이터)
def dataset_load(dataset_url):
    all_data = pd.read_csv(dataset_url, sep='\t')
    all_data = all_data.sample(frac=1).reset_index(drop=True)
    train_data, test_data = dataset_split(dataset=all_data, split_val=0.9)
    train_data, dev_data = dataset_split(dataset=train_data, split_val=0.9)

    return train_data, dev_data, test_data

# 학습을 위한 데이터 셋 전처리 함수
# 입력 : dataset(전처리 대상 데이터셋)
# 출력 : tokens(토큰화 결과), x_data(입력데이터), y_data(정답데이터)
def preprocessing_dataset(dataset):
    tokens, indices, labels = [], [],[]
    
    # 데이터셋의 문장을 토큰화 및 인코딩 처리하고, 라벨을 One hot 벡터 변환으로 처리한다.
    for label, sentence in tqdm(zip(dataset['label'], dataset['sentence']), desc = "데이터 전처리 진행중"):
        tokens.append(tokenizer.tokenize(sentence))
        ids, _ = tokenizer.encode(sentence, max_len=MAX_SEQ_LEN)
        indices.append(ids)

        if label == "A":
            labels.append([1, 0, 0])
        elif label == "B":
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])

    x_data = np.array(indices)
    y_data = np.array(labels)
    print("===> 전처리 결과 출력 <===")
    print("===> tokens sample : ", tokens[0])
    print("===> indices sample : ", indices[0])
    print("===> x_data shape : ", x_data.shape)
    print("===> y_data shape : ", y_data.shape)
    
    return tokens, x_data, y_data

# 메인 프로그램 시작
if __name__ == '__main__':
    # KorPat Tokenier 선언
    tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

    # 필요 데이터셋 로드
    train_data, dev_data, test_data = dataset_load(dataset_path)

    # 모든 데이터셋 전처리
    print("===> 학습데이터 샘플 출력 및 전처리 시작 <===")
    print(train_data[:10])
    train_tokens, train_x, train_y = preprocessing_dataset(train_data)

    print("\n\n===> 검증데이터 샘플 출력 및 전처리 시작 <===")
    print(dev_data[:10])
    dev_tokens, dev_x, dev_y = preprocessing_dataset(dev_data)

    print("\n\n===> 평가데이터 샘플 출력 및 전처리 시작 <===")
    print(test_data[:10])
    test_tokens, test_x, test_y = preprocessing_dataset(test_data)

    # 라벨의 크기 정의
    label_size = train_y.shape[1]
    print("===> 라벨 사이즈 : ", label_size)
    
    # 분류 모델 네트워크 정의, ※ MirroredStrategy (분산처리 포함)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # 입력 레이어 생성
        input_ids  = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')

        # BERT 언어모델 레이어 생성
        bert_params = bert.params_from_pretrained_ckpt(pretreind_model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        # 입력레이어와 BERT 레이어 연결
        bert_output = l_bert(input_ids) 
        print("bert shape", bert_output.shape)

        # BERT 출력 중 Context 정보가 담긴, [CLS] 토큰 정보를 추출
        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)

        # 최종 분류를 위한 클래스 개수로 구성된 Dense 레이어를 연결
        outputs = Dense(units=label_size, activation='softmax')(cls_out)

        # 모델 빌딩
        model = keras.Model(inputs=input_ids, outputs=outputs)
        model.build(input_shape=(None, MAX_SEQ_LEN))

        # BERT 언어모델의 초기 가중치 로드
        bert.load_stock_weights(l_bert, checkpoint_path)

        # 모델 컴파일 과정
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LR),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
    # 모델 요약 출력
    model.summary()
    
    # 모델 학습 시작
    history = model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(dev_x, dev_y)
    )

    # 모델 저장 및 테스트 데이터를 이용한 평가
    model.save(save_model_path)
    print(test_x)
    print(type(test_x))
    print(test_y)
    print(type(test_y))          
    eval_result = model.evaluate(test_x, test_y)

    print('\n===> 평가 결과 출력')
    print("Accuracy : %.4f" % (eval_result[1]))