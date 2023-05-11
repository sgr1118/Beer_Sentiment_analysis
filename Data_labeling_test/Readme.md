## Data Labeling을 테스트한 결과를 저장해둔 폴더입니다.

### 사용된 라이브러리 및 api
- Self_labeling(직접 수행한 라벨링), VADER, Alpaca, GPT API

### 결과 - 1 : 라벨링 분포

|분류|Self_labeling|VADER|Alpaca|GPT API|
|-|-|-|-|-|
|Positive|474|743|346|493|
|Negative|327|181|533|330|
|Neutral|198|76|120|177|

- VADER : 긍정으로 분포가 쏠리는 형태
- Alpack : 중립이 가장 적고 부정 라벨링이 더 많은 형태
- GPT API : 긍정 라벨링이 부정 라벨링보다 더 많고 중립은 Alpaca보다 높은 형태

### 결과 - 2 : self_labeling과 비교하여 일치 여부

|분류|VADER|Alpaca|GPT API|
|-|-|-|-|
|True|605|625|724|
|False|395|375|276|

- self_labeling과 가장 유사한 labeling은 GPT API이다.
- 다만 모든 데이터를 GPT API를 사용하기에는 비용 부담이 과하여 MultinomialNB를 사용하여 전체 데이터에 labeling을 시도하는 방법을 사용
