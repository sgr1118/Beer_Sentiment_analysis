import re
from glob import glob
import pandas as pd

def preprocess_sentence(sentence):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (.) 제거
    sentence = re.sub(r'[.,?!]+[/.,?!]', '', sentence) # 여러개 자음, 모음, 구두점 제거
    sentence = re.sub(r"[^a-z0-9.,!?'""]", " ", sentence) # 지정한 문자 제외 공백으로 전환
    sentence = re.sub(r"[\d\/\d\/\d]", "", sentence) # 날짜 삭제
    sentence = re.sub(r"oz", "", sentence) # oz 삭제
    sentence = re.sub(r"bottle", "", sentence) # bottle 삭제
    sentence = re.sub(r"tab", "", sentence) # tab 삭제
    sentence = re.sub(r'[" "]+', " ", sentence) # 여러개 공백을 하나의 공백으로 바꿉니다.
    sentence = sentence.strip() # 문장 양쪽 공백 제거

    return sentence


df = pd.read_csv('Raw dataset Route')

df['Review'] = df['Review'].apply(lambda x: preprocess_sentence(x))

df = df[df['Review']!='']

df.to_csv('preprocess.csv', index=False)
df.info()