# 택시 요금 예측 프로젝트
이 프로젝트는 Apache Spark와 PySpark를 사용하여 택시 여행 데이터를 분석하고, 여행 거리 및 다른 요인을 바탕으로 택시 요금을 예측하는 모델을 구축합니다. 데이터 처리, 탐색적 데이터 분석(EDA), 선형 회귀 모델을 통해 택시 요금을 예측하는 과정을 포함하고 있습니다.

## 프로젝트 구조

### 파일:
- SQL_analysis.ipynb: 이 스크립트는 Spark SQL을 사용하여 탐색적 데이터 분석(EDA) 및 데이터 정리를 수행합니다. `trip_distance`, `total_amount`, `pickup_time`, `pickup_zone`, `dropoff_zone` 등 여러 특성들 간의 관계를 분석합니다.
- Spark_MLib.py: 이 스크립트는 Spark MLlib을 사용하여 택시 요금을 예측하는 선형 회귀 모델을 구축합니다. 데이터 전처리, 특성 엔지니어링, 모델 학습, 평가 및 학습된 모델의 저장/불러오기 작업이 포함됩니다.

스크립트를 실행하기 전에 아래와 같은 항목들이 설치되어 있어야 합니다:
- Apache Spark
- PySpark
- Python 3.x
- Pandas
- Matplotlib
- Seaborn etc...
