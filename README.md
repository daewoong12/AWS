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
