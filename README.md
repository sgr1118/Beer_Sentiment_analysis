# Project : 맥주 리뷰 감정 분석

## 프로젝트 후원 : (주)모두의 연구소
- 해당 프로젝트는 (주)모두의연구소로부터 지원을 받았음을 알려드립니다.


<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white">

## 💡프로젝트 소개
```
1️⃣ 주제 : RateBeer Review 데이터를 사용한 데이터 분석
2️⃣ 데이터셋 : RateBeer crawling data (자체 수집)
3️⃣ 컬럼 : Rating, Review, Beer_name 
4️⃣ 모델 : SimpleT5
5️⃣ 간단 설명 : 맥주 리뷰 데이터를 사용하여 의미 연결망 분석과 감정 분석 결과 제공
```

## 🏅 프로젝트 목표
#### 1. 데이터 크롤링 (완료)
#### 2. 데이터 시각화 및 EDA (완료) 
#### 3. 데이터 라벨링 (완료)
#### 4. 데이터 전처리 (완료)
#### 5. 감정 분석 (진행중)
#### 6. 개선 사항 적용 (진행 예정)

---
## 전체 프로세스

### 1. 데이터 수집
- Selenium 4.0 기반으로 크롤링 code를 제작하여 Rating, Review, Beer_name 수집

### 2. 시각화 및 EDA
##### Rating 분포와 Review Text의 특이 사항 확인

### 3. 데이터 전처리
- '...'만 작성된 Review 삭제
- english로 작성되지 않은 data 삭제
- 여러 번의 공백을 한번으로 바꾸기
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
- Simple T5 (Pytorch Lightning)
- Pytoch Multi GPU 방식을 사용하여 속도 및 개선된 Mt5, Byt5 사용 (현재 시도중)
---
 
## 🗓️ 프로젝트 진행 일정

|내용|M1|M2|M3|M4|M5|M6|M7|
|---|---|---|---|---|---|---|---|
|데이터 수집|●|●||||||
|EDA 및 시각화||●|●|||||
|데이터 라벨링|||●|||||
|모델링|||||●|●||
|개선사항 적용||||||||
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
---
## 📑 프로젝트 결과물 모음
|No|내용|깃허브|관리 팀원|
|-|-|-|-|
|1|데이터 수집|[📂]()|손기락|
|2|데이터 라벨링|[📂]()|손기락, 하승범|
|3|모델링|[📂]()|손기락, 하승범|
|4|결과|[📂]()|손기락, 하승범|
---
## 🏆 프로젝트 결과(현재 집계중)
|Model|Accuracy|F1-Score(macro)|Precision(macro)|Recall(macro)|
|---|---|---|---|---|
|T5(Alpaca_labeling)|0.757|0.757|0.717|0.721|
|T5(GPT API and MultinomialNB)|0.908|0.747|0.807|0.723|

---
## 🏆 프로젝트 결과

