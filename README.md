# 택시 요금 예측 프로젝트
이 프로젝트는 Apache Spark와 PySpark를 사용하여 택시 여행 데이터를 분석하고, 여행 거리 및 다른 요인을 바탕으로 택시 요금을 예측하는 모델을 구축합니다. 데이터 처리, 탐색적 데이터 분석(EDA), 선형 회귀 모델을 통해 택시 요금을 예측하는 과정을 포함하고 있습니다.

## 프로젝트 구조
### 1. **SQL_analysis.ipynb**
EDA를 수행하고 택시 여행 데이터를 정리합니다. 주요 작업은 다음과 같습니다:
- 여행 데이터와 지역 데이터(픽업/하차 구역)를 로드합니다.
- 기본적인 통계값을 확인하고 결측치를 계산합니다.
- 택시 요금, 이동 거리, 승객 수 등 다양한 변수 간의 관계를 시각화합니다.

### 2. **Spark_MLib.ipynb**
선형 회귀 모델을 사용하여 택시 요금을 예측합니다. 주요 단계는 다음과 같습니다.
- **데이터 로드 및 전처리**: `trip_distance`와 `total_amount` 컬럼을 기반으로 데이터를 정리하고, 불필요한 이상치를 제거합니다.
- **데이터 분할**: 훈련 데이터(train_df)와 테스트 데이터(test_df)로 데이터를 나눕니다.
- **특성 벡터화**: `VectorAssembler`를 사용해 `trip_distance`를 모델에 입력할 수 있는 형태로 변환합니다.
- **모델 학습**: 선형 회귀 모델을 학습시키고, 테스트 데이터에 대한 예측을 수행합니다.
- **모델 평가**: RMSE(평균 제곱근 오차)와 R-squared(결정 계수)를 계산하여 모델의 성능을 평가합니다.
- **모델 저장 및 로드**: 학습된 모델을 저장하고, 나중에 불러와서 예측을 수행할 수 있도록 합니다.

아래와 같은 항목들이 설치되어 있어야 합니다:
- Apache Spark
- PySpark
- Python 3.x
- Pandas
- Matplotlib
- Seaborn etc...

### 데이터 컬럼 설명
- **passenger_count**: 승객 수를 나타내는 정수형 컬럼. 예: `1`, `2`, `3`
- **pickup_location_id**: 택시를 탑승한 지역의 ID. 특정 지역을 나타내는 고유 ID입니다.
- **dropoff_location_id**: 택시에서 하차한 지역의 ID. 특정 하차 지역을 나타내는 고유 ID입니다.
- **trip_distance**: 여행의 총 거리(km 단위). 예: `5.3`, `10.7`
- **pickup_time**: 택시를 탑승한 시간의 "시" 부분. 예: `8`
- **day_of_week**: 택시가 탑승한 요일을 나타내는 문자열. 예: `"Monday"`, `"Friday"`
- **total_amount**: 택시 여행의 총 금액. 예: `15.5`, `25.0`
- **pickup_location_id_idx**: `pickup_location_id`에 대한 숫자형 인덱스 값.
- **pickup_location_id_onehot**: `pickup_location_id_idx`의 원핫 인코딩 결과.
- **dropoff_location_id_idx**: `dropoff_location_id`에 대한 숫자형 인덱스 값.
- **dropoff_location_id_onehot**: `dropoff_location_id_idx`의 원핫 인코딩 결과.
- **day_of_week_idx**: `day_of_week`에 대한 숫자형 인덱스 값.
- **day_of_week_onehot**: `day_of_week_idx`의 원핫 인코딩 결과.
- **passenger_count_vecotr**: `passenger_count`를 벡터 형식으로 변환한 값.
- **passenger_count_scaled**: `passenger_count` 값을 표준화(스케일링)하여 평균 0, 표준편차 1로 변환한 값.
- **trip_distance_vecotr**: `trip_distance`를 벡터 형식으로 변환한 값.
- **trip_distance_scaled**: `trip_distance` 값을 표준화(스케일링)하여 평균 0, 표준편차 1로 변환한 값.
- **pickup_time_vecotr**: `pickup_time`을 벡터 형식으로 변환한 값.
- **pickup_time_scaled**: `pickup_time` 값을 표준화(스케일링)하여 평균 0, 표준편차 1로 변환한 값.
- **feature_vector**: 위에서 설명한 모든 변수를 하나의 벡터로 결합한 값. 이 값은 모델 학습에 사용됩니다.

### 3. **Airflow**
스파크로 구현한 모든 과정을 자동화할 수 있도록 에어플로우 테스크로 구현하였다.
#### 3.1 **preprocess.py**
데이터를 전처리하는 작업을 수행 -> 데이터 전처리를 통해 모델 학습에 적합한 형태로 데이터를 준비
- SparkSession을 사용하여 Spark 환경을 설정하고 데이터를 로드합니다.
- SQL 쿼리를 통해 데이터를 필터링하고 필요한 컬럼(passenger_count, pickup_location_id, dropoff_location_id, trip_distance, pickup_time, day_of_week, total_amount)을 선택합니다.
- 데이터 범위를 2021년 1월 1일부터 2021년 8월 1일까지로 제한하고, 잘못된 데이터(예: total_amount가 5000 이상, trip_distance가 500 이상인 경우 등)를 제거합니다.
- 데이터를 학습용(train_df)과 테스트용(test_df)으로 나누고, parquet 형식으로 저장합니다.

#### 3.2 **tune_hyperarameter.py**
Airflow DAG를 정의하며, 데이터 파이프라인의 흐름을 관리 -> Airflow를 사용해 데이터 처리, 하이퍼파라미터 튜닝, 모델 학습 등을 자동화
- Airflow의 DAG 객체를 정의하여, preprocess.py, tune_hyperparameter.py, train_model.py 순서대로 실행됩니다.
- SparkSubmitOperator를 사용하여 Spark 애플리케이션을 실행합니다.
- 각 스텝(preprocess, tune_hyperparameter, train_model)을 순차적으로 실행하고, 파이프라인의 흐름을 관리합니다.
  
#### 3.3 **train_model.py**
머신러닝 모델을 학습하는 작업을 수행 -> LinearRegression 모델을 학습하고, 학습된 모델을 저장하여 이후 예측에 사용할 수 있게 한다.
- 데이터를 로드한 후, 범주형 변수는 StringIndexer와 OneHotEncoder를 사용해 처리하고, 수치형 변수는 VectorAssembler와 StandardScaler를 사용해 벡터화 및 스케일링합니다.
- LinearRegression 모델을 학습하고, 예측 결과를 저장합니다.
- 학습된 모델을 지정된 디렉토리에 저장합니다.

#### 3.4 **taxi-price-pipeline.py**
하이퍼 파라미터 튜닝을 위한 작업을 수행 -> 모델 학습에 최적의 하이퍼파라미터를 찾기 위해 하이퍼파라미터 튜닝을 수행
- 데이터를 샘플링하여 학습용 데이터를 준비합니다.
- StringIndexer와 OneHotEncoder, VectorAssembler, StandardScaler 등을 사용하여 데이터 전처리를 수행합니다.
- CrossValidator와 ParamGridBuilder를 사용해 하이퍼파라미터(elasticNetParam, regParam)를 튜닝합니다.
- 최적의 하이퍼파라미터를 찾아 hyperparameter.csv 파일로 저장합니다.
