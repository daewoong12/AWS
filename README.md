# 택시 요금 예측 프로젝트
- 이 프로젝트는 Apache Spark와 PySpark를 사용하여 택시 여행 데이터(2021년 01월 ~ 07월)를 분석하고, 여행 거리 및 다른 요인 등을 바탕으로 택시 요금을 예측하는 모델을 구축합니다. 데이터 처리, 탐색적 데이터 분석(EDA), 선형 회귀 모델을 통해 택시 요금을 프로세싱하는 과정을 포함하고 있습니다.

! 아래와 같은 항목들이 설치되어 있어야 합니다.
- Apache Spark
- PySpark
- Python 3.8.20
- Pandas
- Matplotlib
- Seaborn
- apache-airflow
- apache-airflow-providers-apache-spark etc...


# 프로젝트 구조
- jupyter notebook(SQL_analysis.ipynb, Spark_MLib.ipynb)을 활용해서 전처리, EDA, 예측 모델 학습 및 약간의 분석보고서를 작성. 
- Airflow를 사용하여 위에 했던 작업들(데이터 전처리, 하이퍼파라미터 튜닝, 모델 학습)의 모든 작업을 자동화하였습니다. Airflow의 **DAG**를 정의하여 각 작업을 순차적으로 실행하며, 이를 통해 데이터 파이프라인을 관리.
![image](https://github.com/user-attachments/assets/a7974ce2-49a0-482f-9c82-61a2ed8b5609)
### 데이터 컬럼 설명
- **passenger_count**: 승객 수를 나타내는 정수형 컬럼. 예: `1`, `2`, `3`
- **pickup_location_id**: 택시를 탑승한 지역의 ID. 특정 지역을 나타내는 고유 ID.
- **dropoff_location_id**: 택시에서 하차한 지역의 ID. 특정 하차 지역을 나타내는 고유 ID.
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
- **feature_vector**: 위에서 설명한 모든 변수를 하나의 벡터로 결합한 값. 이 값은 모델 학습에 사용.

## 1. **SQL_analysis.ipynb**
EDA를 수행하고 택시 여행 데이터를 정리합니다. 작업 단계는 다음과 같습니다
- 여행 데이터(yellow_tripdata_2021-01~07.csv)와 지역 데이터(taxi+_zone_lookup.csv)를 로드합니다.
- 기본적인 통계값을 확인하고 데이터의 결측치와 이상치를 분석해서 불필요한 데이터는 없앱니다.
    ![image](https://github.com/user-attachments/assets/044e82d9-6aff-4de0-974a-7ebbe30a33e1)
1. c.total_amount < 5000 AND c.total_amount > 0
- total_amount는 택시 요금인데, 이상치가 존재하는 것을 발견했습니다. 예를 들어, 매우 높은 요금(5000 이상)은 실수나 데이터 오류일 가능성이 크고, 0 이하의 값도 실제 요금으로 존재하지 않기 때문에 이를 제외했습니다.

2. c.trip_distance < 100
- trip_distance는 여행 거리입니다. 택시 요금 예측 모델에서는 너무 긴 거리(100km 이상)는 현실적으로 드물고, 이 역시 이상치로 간주될 수 있습니다. 따라서 이 값을 필터링하여 모델 학습에 불필요한 데이터를 제거합니다.

3. c.passenger_count < 5
- passenger_count는 탑승 승객 수를 나타내는 변수입니다. 대부분의 택시 여행은 1~4명의 승객이 탑승하므로, 5명 이상의 승객이 탑승한 경우는 이상치일 가능성이 높습니다. 이 범위를 초과한 데이터를 필터링하여 예측 모델의 정확성을 높입니다.

4. c.pickup_date >= 2021-01-01' AND c.pickup_date < '2021-08-01
- pickup_date는 택시가 승객을 탑승시킨 날짜를 나타냅니다. 이 필터는 2021년 1월 1일부터 2021년 8월 1일까지의 데이터를 사용하도록 제한하여, 학습 데이터의 시점을 일정 범위로 제한하고 있습니다. 이 범위 밖의 데이터는 예측에 불필요할 수 있으므로 제외합니다.

- 택시 요금, 이동 거리, 승객 수 등 다양한 변수 간의 관계를 시각화합니다.


## 2. **Spark_MLib.ipynb**
위에서 작업했던 데이터를 활용하여, 선형 회귀 모델을 사용하여 택시 요금을 예측합니다. 작업 단계는 다음과 같습니다.
- **데이터 분할**: 훈련 데이터(train_df)와 테스트 데이터(test_df)로 데이터를 나눕니다.
![image](https://github.com/user-attachments/assets/4a80ccc6-3143-44f7-8b98-d7797b3d0d5b)
- **특성 벡터화**: `VectorAssembler`를 사용해 `trip_distance`를 모델에 입력할 수 있는 형태로 변환합니다.
- **모델 학습**: 선형 회귀 모델을 학습시키고, 테스트 데이터에 대한 예측을 수행합니다.
- **모델 평가**: RMSE(평균 제곱근 오차)와 R-squared(결정 계수)를 계산하여 모델의 성능을 평가합니다.
- **모델 저장 및 로드**: 학습된 모델을 저장하고, 나중에 불러와서 예측을 수행할 수 있도록 합니다.

#### 모델 예측 : R² = 0.7966, RMSE = 5.8801
- 모델은 trip_distance(여행 거리)를 입력으로 사용하여 total_amount(택시 요금)을 예측. trip_distance 값을 알고 있을 때, total_amount 택시 요금을 예측할 수 있다는 의미.
- RMSE(오차)는 예측값과 실제값 간의 차이를 나타내는 지표로, 값이 작을수록 모델이 예측을 잘 수행하고 있다는 것을 의미합니다. 예를 들어, RMSE(오차)가 5.88이라는 것은 모델이 예측한 값과 실제 total_amount 값 간에 평균적으로 약 5.88의 차이가 있다는 뜻입니다.
- R²(결정계수)는 모델이 trip_distance와 total_amount 간의 관계를 얼마나 잘 설명하는지를 나타내며, 약 79.66%의 설명력을 가지고 있습니다.

## 3. **Airflow**
스파크로 구현한 모든 과정을 자동화할 수 있도록 Airflow로 구현하였다.
### 보안그룹 설정
![image](https://github.com/user-attachments/assets/96d5c7e3-3474-4215-82e4-6e0e7fa65f86)
![image](https://github.com/user-attachments/assets/04306a2d-a32d-489d-990f-750aa87c04a1)
![image](https://github.com/user-attachments/assets/5d899f9c-6284-4bf1-893e-52adefde6465)

### 테스크(Task) 작성
- DAG를 구성하기 앞서, Airflow(자동화)로 구성할 테스크들을 작성한다.
** DAG란 ** 워크플로우를 정의하는 코드 구조로, 각 작업(Task)은 DAG 내에서 실행되어야 할 순서를 지정하며, DAG를 통해 전체 파이프라인을 관리할 수 있습니다.
- 앞의 머신러닝 절차를 파이썬 파일로 생성할 것이다.
  
#### 3.1 **preprocess.py**
데이터 전처리를 통해 모델 학습에 적합한 형태로 데이터를 준비.
- SparkSession을 사용하여 Spark 환경을 설정하고 데이터를 로드합니다.
- SQL 쿼리를 통해 데이터를 필터링하고 필요한 컬럼(passenger_count, pickup_location_id, dropoff_location_id, trip_distance, pickup_time, day_of_week, total_amount)을 선택합니다.
- 데이터 범위를 2021년 1월 1일부터 2021년 8월 1일까지로 제한하고, 불필요한 데이터를 제거합니다.
- 데이터를 학습용(train_df)과 테스트용(test_df)으로 나누고, parquet 형식으로 저장합니다.

#### 3.2 **tune_hyperarameter.py**
모델 학습에 최적의 하이퍼파라미터를 찾기 위해 하이퍼파라미터 튜닝을 수행
- 데이터를 샘플링하여 학습용 데이터를 준비합니다.
- 범주형 변수는 StringIndexer와 OneHotEncoder를 사용해 처리하고, 수치형 변수는 VectorAssembler와 StandardScaler를 사용해 벡터화 및 스케일링합니다.
![image](https://github.com/user-attachments/assets/fcc29698-1bc1-468b-9702-3fc5a7291e04)

- CrossValidator와 ParamGridBuilder를 사용해 하이퍼 파라미터(elasticNetParam, regParam)를 튜닝합니다.
![image](https://github.com/user-attachments/assets/6a46e295-9938-4ba1-829a-fd935278a0f7)
1. CrossValidator: 교차 검증(Cross-validation)을 수행하는 객체로, 주어진 데이터셋을 여러 개의 폴드(fold)로 나누고 각 폴드에 대해 모델을 학습하고 평가합니다. 교차 검증은 모델이 훈련 데이터에 과적합(overfitting)되지 않도록 하여, 모델의 일반화 성능을 평가할 수 있습니다.

2. ParamGridBuilder: 하이퍼파라미터의 값을 여러 가지 조합으로 설정할 수 있도록 돕는 객체입니다. 다양한 하이퍼파라미터 값을 정의한 후, CrossValidator에 제공하여 모델을 평가할 수 있습니다.

3. elasticNetParam: 엘라스틱 넷(Elastic Net) 모델에서 사용되는 하이퍼파라미터입니다. 이 파라미터는 모델의 복잡도를 제어하는 중요한 역할을 합니다.

4. regParam: 정규화 파라미터로, 모델의 과적합을 방지하기 위해 사용됩니다.

- 최적의 하이퍼파라미터를 찾아 hyperparameter.csv 파일로 저장합니다.
  
#### 3.3 **train_model.py**
머신러닝 모델을 학습하는 작업을 수행 -> LinearRegression 모델을 학습하고, 학습된 모델을 저장하여 이후 예측에 사용할 수 있게 한다.
- LinearRegression 모델을 학습하고, 예측 결과를 저장합니다.
- 학습된 모델을 지정된 디렉토리에 저장합니다.

#### 3.4 **taxi-price-pipeline.py**
Airflow DAG를 정의하며, 데이터 파이프라인의 흐름을 관리 -> Airflow를 사용해 데이터 처리, 하이퍼파라미터 튜닝, 모델 학습 등을 자동화
- Airflow의 DAG 객체를 정의하여, preprocess.py, tune_hyperparameter.py, train_model.py 순서대로 실행됩니다.
- SparkSubmitOperator를 사용하여 Spark 애플리케이션을 실행합니다.
- 각 스텝(preprocess, tune_hyperparameter, train_model)을 순차적으로 실행하고, 파이프라인의 흐름을 관리합니다.
![image](https://github.com/user-attachments/assets/d88f0190-d6b5-4508-87ef-d64a2bda9342)


task 작성을 마쳤으면, <br>
![image](https://github.com/user-attachments/assets/561b86e9-e822-45bf-aa6c-e7272b3c764f)<br>
![image](https://github.com/user-attachments/assets/ac3a103a-a848-4fcd-a2c3-29c9b127678a)<br>
Airflow의 웹 서버를 열고 scheduler를 실행한다.(커맨드창 두개를 띄우는 것을 추천!)

url : (내 EIP:8080/)
![image](https://github.com/user-attachments/assets/a1e17897-1a10-45f8-9ac9-d6c90e1f2849)<br>
![image](https://github.com/user-attachments/assets/8cbfdf53-109c-4bd2-86f9-014e08d494b7)<br>
![image](https://github.com/user-attachments/assets/5c62a8f9-0d76-496f-aba3-cdd455b8efe0)<br>
위와 같이 저장후, DAGs에서 내가 생성한 Dag 파일(taxi-price-pipeline) 클릭<br>
![image](https://github.com/user-attachments/assets/94622cd4-3b27-46f5-8f17-dfc3714d21fc)<br>
![image](https://github.com/user-attachments/assets/c88bc3f1-5a21-4d7e-8ca3-38a8205ef976)<br>
![image](https://github.com/user-attachments/assets/5dea8399-75d4-4071-bf08-08479cb0800c)<br>
위 이미지와 같이 모든 테스크가 정상적으로 진행됨을 확인할 수 있다.<br>

### airflow 예측 결과 해석
예측값과 실제값을 비교하여 모델이 얼마나 정확하게 예측했는지를 확인할 수 있다.
- trip_distance: 실제 여행 거리 값. 모델의 입력 변수로 사용된 값.
- day_of_week: 여행이 발생한 요일을 나타낸다. 이는 범주형 변수로 처리되어 OneHotEncoder로 변환된다.
- total_amount: 실제 택시 요금. 이는 모델이 예측하려는 대상 값
- prediction: 모델이 예측한 택시 요금. trip_distance와 day_of_week를 기반으로 예측된 값.
![image](https://github.com/user-attachments/assets/f071c1ce-f785-4921-99b9-4f167d42dc29)

# 느낀점
- kafka를 활용해서 실시간으로 데이터가 들어오는 것을 처리해서 Airflow에 적용시키는 것을 시도를 해봤지만 시간이 부족해서 못구현한게 너무 아쉽다.
- airflow와 kafka에 집중하다 보니, mysql에 적재 불러오기 기능을 넣는 것을 까먹었다.
- 모델 예측 성능도 오차 5.8, 예측률 약 79%라는 안좋은 성능을 보였는데 이것도 다른 작업들을 하느라 개선을 못한 것이 아쉽다.
- 데이터가 너무 많아서 한번 실행시키는데 시간이 오래걸렸다.
