# 필요 라이브러리 임포트
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tqdm import tqdm
from korpat_tokenizer import Tokenizer
import tensorflow as tf
import bert
import os
import mecab
import pandas as pd
import numpy as np

model = load_model("./korpat_bert_test_classification_model.h5",custom_objects={"BertModelLayer": bert.BertModelLayer})


#분류하고자 하는 파일 설정
dataset_path = "./classification_test.csv"              # 분류를 원하시는 dataset의 경로를 넣어주시면 됩니다.

# 필요 환경변수 설정
os.environ['TF_KERAS'] = '1'    # Keras Tensorflow 설정
config_path     = "./pretrained/korpat_bert_config.json" # KorpatBert Config 파일 경로
vocab_path      = "./pretrained/korpat_vocab.txt"        # KorpatTokenizer Vocabulary 파일 경로
checkpoint_path = "./pretrained/model.ckpt-381250"       # KorpatBert 모델파일 경로
pretreind_model_dir = "./pretrained/"                    # KorpatBert 모델 디렉토리 경로


MAX_SEQ_LEN = 512 # 학습 최대 토큰 갯수


# 데이터 셋 로드 함수
# 입력 : dataset_url (전체 데이터셋 경로)
# 출력 : test_data(평가데이터)
def dataset_load(dataset_url):
    test_data = pd.read_csv(dataset_url,encoding='cp949')
    return test_data

# 학습을 위한 데이터 셋 전처리 함수
# 입력 : dataset(전처리 대상 데이터셋)
# 출력 : tokens(토큰화 결과), x_data(입력데이터), y_data(정답데이터)
def preprocessing_dataset(dataset):
    tokens, indices, labels = [], [],[]
    
    # 데이터셋의 문장을 토큰화 및 인코딩 처리하고, 라벨을 One hot 벡터 변환으로 처리한다.
    for sentence, label in tqdm(zip(dataset['sentence'],dataset['label']), desc = "데이터 전처리 진행중"):
        tokens.append(tokenizer.tokenize(sentence))
        ids, _ = tokenizer.encode(sentence, max_len=MAX_SEQ_LEN)
        indices.append(ids)
    x_data = np.array(indices)
        
    return tokens, x_data


# 메인 프로그램 시작
if __name__ == '__main__':
    # KorPat Tokenier 선언
    tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

    # 필요 데이터셋 로드
    test_data = dataset_load(dataset_path)
    print("\n\n===> 평가데이터 샘플 출력 및 전처리 시작 <===")
    print(test_data[:10])
    test_tokens, test_x  = preprocessing_dataset(test_data)
    print("test_x",test_x)
    print("test_x",len(test_x))   
      
    # 라벨의 크기 정의
    # 모델 저장 및 테스트 데이터를 이용한 평가
    predict = model.predict(test_x) #id 
    print("text_x예측embedding",predict)
    preds_flat = np.argmax(predict, axis=1).flatten()
    print("예측값",preds_flat)
    print("text_x예측개수",len(preds_flat))
    #모델이 예측한 값을 저장하다.
      
    print('\n===> 분류가 완료 되었습니다')
    #df=pd.concat([test_data['sentence'],preds_flat])
    df=pd.DataFrame({'원문sentence':test_data['sentence'],'예측값':preds_flat})
    df.to_csv(f"./classification_results.csv",encoding='cp949')      