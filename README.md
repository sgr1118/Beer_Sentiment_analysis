# Project : 맥주 리뷰 감정 분석

## 프로젝트 후원 : (주)모두의연구소, K-디지털 플랫폼
---
본 프로젝트는 모두의연구소와 K-디지털 플랫폼으로부터 지원받았습니다.

<img src="https://img.shields.io/badge/Python-3.8-blue"><img src="https://img.shields.io/badge/Pytorc%20Lhlightning-1.5.10-blue"><img src="https://img.shields.io/badge/Transformers-4.16.2-blue"><img src="https://img.shields.io/badge/-Colab-yellow)">

## 감정 분석 결과 코드 예시
```c
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F

model_path = '/home/beerlab/outputs/simplet5-epoch-2-train-loss-0.0396-val-loss-0.053'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

def analyze_sentiment(sentence):
    input_text = "sentiment: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_embeds = model.get_input_embeddings()(input_ids)

    # T5 모델을 사용하여 감정 분석 수행
    output = model.generate(input_ids=input_ids, decoder_inputs_embeds=input_embeds, max_length=200)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # 확률을 얻기 위해 로짓 계산
    logits = model(input_ids=input_ids, decoder_inputs_embeds=input_embeds).logits
    probabilities = F.softmax(logits, dim=-1).squeeze()

    sentiment = decoded_output
    emotions = ["Positive", "Negative"]
    emotion_probabilities = probabilities[0].tolist()  # 첫 번째 원소를 추출하여 확률값으로 사용
    result = {"sentiment": sentiment, "emotions": dict(zip(emotions, emotion_probabilities))}
    return result

sentences = ['I took a sip and immediately discarded it. How could a beer have such a strong cinnamon flavor?',
             'The taste of this beer embodies the style description quite well.',
             "This beer successfully captures the bitter aroma of coffee, and when you're about to finish a glass, there's a slight dominant sweetness that lingers in your mouth."]
for sentence in sentences:
    predicted_sentiment = analyze_sentiment(sentence)
    print("문장:", sentence)
    print("감정:", predicted_sentiment["sentiment"])
    
문장: I took a sip and immediately discarded it. How could a beer have such a strong cinnamon flavor?
감정: Negative
문장: The taste of this beer embodies the style description quite well.
감정: Positive
문장: This beer successfully captures the bitter aroma of coffee, and when you're about to finish a glass, there's a slight dominant sweetness that lingers in your mouth.
감정: Positive
```

## 💡프로젝트 소개
```
1️⃣ 주제 : RateBeer Review 데이터를 사용한 데이터 분석
2️⃣ 데이터셋 : RateBeer crawling data (자체 수집)
3️⃣ 컬럼 : Rating, Review, Beer_name 
4️⃣ 모델 : SimpleT5(T5)
5️⃣ 간단 설명 : 맥주 리뷰 데이터를 사용하여 감정 분석 결과 제공
6️⃣ 기대 효과 : 상품에 대한 감정 분석을 기반으로 제품 기획 및 마케팅 활용에 근거로 사용
```

## 🏅 프로젝트 목표
#### 1. 데이터 크롤링 (완료)
#### 2. 데이터 시각화 및 EDA (완료) 
#### 3. 데이터 라벨링 (완료)
#### 4. 데이터 전처리 (완료)
#### 5. 감정 분석 (완료)
#### 6. 개선 사항 적용 (완료)

---
## 전체 프로세스

### 1. 데이터 수집
- Selenium 4.0 기반으로 크롤링 code를 제작하여 Rating, Review, Beer_name 수집
- XPATH 경로를 통하여 Elements를 수집한다.

### 2. 시각화 및 EDA
##### Rating 분포와 Review Text의 특이 사항 확인

### 3. 데이터 전처리
- '...'만 작성된 Review 삭제
- english로 작성되지 않은 data 삭제
- 여러 번의 공백을 한번으로 바꾸기
- 불용어 제거
- 지나치게 긴 데이터 삭제 (len >= 300)
- VADER, Alpaca, GPT API and MultinomialNB을 사용하여 Labeling 수행하고 self_labeling과 비교

<Sample Data 적용한 Labeling - self_labeling과 비교>
|라이브러리|self_labeling|VADER|Alpaca|GPT API|
|-|-|-|-|-|
|Positive|474|743|346|493|
|Negative|327|181|533|330|
|Neutral|198|76|120|177|

<전체 Data 적용한 Labeling>
|라이브러리|VADER|Alpaca|GPT API and MultinomialNB|
|-|-|-|-|
|Positive|84732|36897|64021|
|Negative|19306|55934|40520|
|Neutral|8768|19094|3400|

### 4. 감정 분석
- Pytoch Multi GPU 적용하여 Simple T5 (Pytorch Lightning) 학습 수행

### 5. 개선 사항
- [Data Augmentation 기법 적용](https://maelfabien.github.io/machinelearning/NLP_8/#when-should-we-use-data-augmentation)
- SR : 동의어 교체, RD : 무작위 삭제, RS : 무작위 교체, RI : 무작위 삽입
- 동의어 교체 방식이 가장 높은 성능을 보여주었다.
---
 
## 🗓️ 프로젝트 진행 일정

|내용|M1|M2|M3|M4|M5|M6|M7|
|---|---|---|---|---|---|---|---|
|데이터 수집|●|●||||||
|EDA 및 시각화||●|●|||||
|데이터 라벨링|||●|||||
|모델링||||●|●|●||
|개선사항 적용||||||●|●|
---
## 🦄 프로젝트를 위한 자료
#### [1. Hugging_Face_T5_Guide](https://huggingface.co/docs/transformers/model_doc/t5)
#### [2. T5_Paper](https://arxiv.org/pdf/1910.10683v3.pdf)
#### [3. SimpleT5 github](https://github.com/Shivanandroy/simpleT5/tree/main)
#### [4. Data_Labeling_VADER](https://medium.com/analytics-vidhya/sentiment-analysis-with-vader-label-the-unlabeled-data-8dd785225166)
#### [5. Data_Labeling_GPT](https://towardsdatascience.com/can-chatgpt-compete-with-domain-specific-sentiment-analysis-machine-learning-models-cdcd9937b460)
#### [6. Data_Labeling_Alpaca](https://www.youtube.com/watch?v=JzBR8oieyy8&t=117s)
#### [7. The Most Common Evaluation Metrics In NLP](https://medium.com/towards-data-science/the-most-common-evaluation-metrics-in-nlp-ced6a763ac8b)
#### [8. Pytorch Multi GPU](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)
#### [9. Data Augmentation 기법](https://maelfabien.github.io/machinelearning/NLP_8/#when-should-we-use-data-augmentation)
---
## 📑 프로젝트 결과물 모음
|No|내용|깃허브|관리 팀원|
|-|-|-|-|
|1|데이터 수집|[📂](https://github.com/sgr1118/Beer_Sentiment_anlysis/tree/main/Ratebeer_Crawling)|손기락|
|2|데이터 라벨링|[📂](https://github.com/sgr1118/Beer_Sentiment_anlysis/tree/main/Data_labeling_test)|손기락, 하승범|
|3|모델링|[📂](https://github.com/sgr1118/Beer_Sentiment_anlysis/tree/main/Sentiment_analsis_result)|손기락, 하승범|
|4|결과|[📂](https://github.com/sgr1118/Beer_Sentiment_analysis/tree/main/Sentiment_prediction)|손기락|
---
## 📑 학습 결과 기록
|Model|Accuracy|F1-Score(macro)|Precision(macro)|Recall(macro)|
|---|---|---|---|---|
|T5(Alpaca_labeling)|0.757|0.710|0.733|0.698|
|T5(GPT API and MultinomialNB)|0.908|0.747|0.807|0.723|
|T5(GPT API and MultinomialNB) - RI(무작위 삽입)|0.904|0.748|0.749|0.747|
|T5(GPT API and MultinomialNB) - RS(무작위 교체)|0.914|0.817|0.740|0.767|
|T5(GPT API and MultinomialNB) - SR(동의어 교체)|0.915|0.767|0.817|0.740|
|T5(GPT API and MultinomialNB) - RD(무작위 삭제)|0.866|0.711|0.772|0.682|

---
## 📑 최종 학습 결과 기록

### Binary Class 분류 결과 
|Model|Accuracy|F1-Score(macro)|Precision(macro)|Recall(macro)|
|---|---|---|---|---|
|T5(GPT API and MultinomialNB) - SR(동의어 교체)|0.946|0.946|0.946|0.946|

### Multi Class 분류 결과 
|Model|Accuracy|F1-Score(wigthed)|Precision(wigthed)|Recall(wigthed)|
|---|---|---|---|---|
|T5(Alpaca_labeling) - SR, RD|0.915|0.767|0.817|0.740|

---
## 📑 프로젝트 개선 요구 사항

### 1. 맥주의 종류에 따른 감정 분석 기준 세부화
- 예를 들어 라거라는 맥주 종류 중 헬레스라는 것이 있다. 헬레스는 기본적으로 매우 목넘김이 너무 편해 'watery'하다고 할 수 있다. 
다만 이런 표현이 특징이 뚜렷한 맥주에 대한 리뷰에서는 부정적으로 받아들여질 가능성이 있다. 그러므로 맥주에 특성을 반영하여 감정을 분류할 필요가 있다.

### 2. 중립적인 리뷰 구분
- 맥주 리뷰에서 단순히 상품을 설명하는 내용이거나 감정이 크게 나타나지않아 중립 라벨링 처리가 필요할 것으로 생각한다.
