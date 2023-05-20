import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) # 이 부분은 사용 여부에 따라 주석 처리를 하길 바람
from nltk.corpus import wordnet
import random

df = pd.read_csv('데이터 경로')
df_sr_sentiment = df[df['label'] == '추출할 감정']

def get_synonyms(word): # 문자에서 알파벳과 공백을 남긴 채 제거
    """
    Get synonyms of a word
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms: 
        synonyms.remove(word)
    return list(synonyms)

def random_insertion(words, n): # 단어의 동의어를 임의의 위치에 무작위로 삽입
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def add_word(new_words):
    
    synonyms = []
    counter = 0
    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
        
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

# 'Review' 열의 각 행에 대해 Synonym Replacement를 수행하는 함수를 apply
df_sr_sentiment['Review'] = df_sr_sentiment['Review'].apply(lambda x: synonym_replacement(x, 3))

# 원래 데이터와 결합
df_sr_sentiment.reset_index(inplace=True)

df_concat = pd.concat([df, df_sr_sentiment], axis = 0) # 행 방향으로 데이터 결합