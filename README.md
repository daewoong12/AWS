# 택시 요금 예측 프로젝트
이 프로젝트는 Apache Spark와 PySpark를 사용하여 택시 여행 데이터를 분석하고, 여행 거리 및 다른 요인을 바탕으로 택시 요금을 예측하는 모델을 구축합니다. 데이터 처리, 탐색적 데이터 분석(EDA), 선형 회귀 모델을 통해 택시 요금을 예측하는 과정을 포함하고 있습니다.

## 프로젝트 구조

### 파일:
- SQL_analysis.ipynb: Spark SQL을 사용하여 탐색적 데이터 분석(EDA) 및 데이터 정리를 수행합니다. `trip_distance`, `total_amount`, `pickup_time`, `pickup_zone`, `dropoff_zone` 등 여러 특성들 간의 관계를 분석합니다.
- Spark_MLib.ipynb: Spark MLlib을 사용하여 택시 요금을 예측하는 선형 회귀 모델을 구축합니다. 데이터 전처리, 특성 엔지니어링, 모델 학습, 평가 및 학습된 모델의 저장/불러오기 작업이 포함됩니다.

스크립트를 실행하기 전에 아래와 같은 항목들이 설치되어 있어야 합니다:
- Apache Spark
- PySpark
- Python 3.x
- Pandas
- Matplotlib
- Seaborn etc...

 |-- passenger_count: integer (nullable = true) :승객 수를 나타내는 정수형 컬럼
 |-- pickup_location_id: integer (nullable = true) : 택시를 탑승한 지역의 ID
 |-- dropoff_location_id: integer (nullable = true) : 택시에서 하차한 지역의 ID
 |-- trip_distance: double (nullable = true) : 여행의 총 거리(km단)
 |-- pickup_time: integer (nullable = true) : 택시를 탑승한 시간의 "시" 부분
 |-- day_of_week: string (nullable = true) : 택시가 탑승한 요일을 나타내는 문자열
 |-- total_amount: double (nullable = true) : 택시 여행의 총 금액
 |-- pickup_location_id_idx: double (nullable = false) : pickup_location_id에 대한 숫자형 인덱스 값
 |-- pickup_location_id_onehot: vector (nullable = true) : pickup_location_id_idx의 원핫 인코딩 결과
 |-- dropoff_location_id_idx: double (nullable = false) : dropoff_location_id에 대한 숫자형 인덱스 값
 |-- dropoff_location_id_onehot: vector (nullable = true) : dropoff_location_id_idx의 원핫 인코딩 결과
 |-- day_of_week_idx: double (nullable = false) : day_of_week에 대한 숫자형 인덱스 값
 |-- day_of_week_onehot: vector (nullable = true) : day_of_week_idx의 원핫 인코딩 결과
 |-- passenger_count_vecotr: vector (nullable = true) : passenger_count를 벡터 형식으로 변환한 값
 |-- passenger_count_scaled: vector (nullable = true) : passenger_count 값을 표준화(스케일링)하여 평균 0, 표준편차 1로 변환한 값
 |-- trip_distance_vecotr: vector (nullable = true) : trip_distance를 벡터 형식으로 변환한 값
 |-- trip_distance_scaled: vector (nullable = true) : trip_distance 값을 표준화(스케일링)하여 평균 0, 표준편차 1로 변환한 값
 |-- pickup_time_vecotr: vector (nullable = true) : pickup_time을 벡터 형식으로 변환한 값
 |-- pickup_time_scaled: vector (nullable = true) : pickup_time 값을 표준화(스케일링)하여 평균 0, 표준편차 1로 변환한 값
 |-- feature_vector: vector (nullable = true) : 위에서 설명한 모든 변수를 하나의 벡터로 결합한 값
