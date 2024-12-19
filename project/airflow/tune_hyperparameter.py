from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import pandas as pd

MAX_MEMORY="5g"
spark = SparkSession.builder.appName("taxi-fare-prediciton")\
                .config("spark.executor.memory", MAX_MEMORY)\
                .config("spark.driver.memory", MAX_MEMORY)\
                .getOrCreate()

data_dir = "/home/lab01/src/project/taxi"

# 데이터 읽기
train_df = spark.read.parquet(f"{data_dir}/train/")

# 샘플 데이터 생성
# toy_df는 train_df에서 10% 샘플을 무작위로 선택한 데이터
# seed=1을 사용하여 재현 가능한 결과를 얻는
toy_df = train_df.sample(False, 0.1, seed=1)

cat_feats = [
    "pickup_location_id",
    "dropoff_location_id",
    "day_of_week"
]

stages = []

# StringIndexer: 각 범주형 변수(pickup_location_id, dropoff_location_id, day_of_week)를 숫자 인덱스로 변환
# OneHotEncoder: StringIndexer로 변환된 인덱스를 원핫 인코딩하여 범주형 변수를 처리
# setHandleInvalid("keep")
for c in cat_feats:
    cat_indexer = StringIndexer(inputCol=c, outputCol= c + "_idx").setHandleInvalid("keep")
    onehot_encoder = OneHotEncoder(inputCols=[cat_indexer.getOutputCol()], outputCols=[c + "_onehot"])
    stages += [cat_indexer, onehot_encoder]

num_feats = [
    "passenger_count",
    "trip_distance",
    "pickup_time"
]

# VectorAssembler: 수치형 변수(passenger_count, trip_distance, pickup_time)를 벡터 형태로 변환
# StandardScaler: 각 수치형 변수의 스케일링
for n in num_feats:
    num_assembler = VectorAssembler(inputCols=[n], outputCol= n + "_vecotr")
    num_scaler = StandardScaler(inputCol=num_assembler.getOutputCol(), outputCol= n + "_scaled")
    stages += [num_assembler, num_scaler]

# assembler_inputs: 원핫 인코딩된 범주형 변수들과 스케일링된 수치형 변수들을 모두 합쳐 특성 벡터(feature_vector)를 만듬
# VectorAssembler: 여러 개의 특성들을 하나의 벡터로 결합하여 모델 학습에 사용할 특성 벡터를 생성
assembler_inputs = [c + "_onehot" for c in cat_feats] + [n + "_scaled" for n in num_feats]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="feature_vector")
stages += [assembler]

# 선형 회귀 모델(LinearRegression)을 정의합니다:
# maxIter=30: 최대 반복 횟수 설정.
# solver="normal": 일반적인 정규 방정식을 사용하여 모델을 학습합니다.
# labelCol='total_amount': 목표 변수인 total_amount를 지정합니다.
# featuresCol='feature_vector': 특성 벡터를 지정
lr = LinearRegression(
    maxIter=30,
    solver="normal",
    labelCol='total_amount',
    featuresCol='feature_vector'
)

# 하이퍼 파라미터 튜닝을 위한 교차 검증(CrossValidator, ParamGridBuilder 사용)
# elasticNetParam(ElasticNet 하이브리드 모델의 비율)과 regParam(정규화 파라미터)을 튜닝하기 위해 가능한 값을 param_grid에 정의
cv_stages = stages + [lr]
cv_pipeline = Pipeline(stages=cv_stages)
param_grid = ParamGridBuilder()\
                .addGrid(lr.elasticNetParam, [0.3, 0.5])\
                .addGrid(lr.regParam, [0.03, 0.05])\
                .build()

cross_val = CrossValidator(estimator=cv_pipeline,
                           estimatorParamMaps=param_grid,
                           evaluator=RegressionEvaluator(labelCol="total_amount"),
                           numFolds=5)


# 모델 학습: 교차 검증을 통해 최적의 하이퍼파라미터를 찾고, cv_model.bestModel에서 최적의 alpha와 reg_param 값을 추출
cv_model = cross_val.fit(toy_df)
alpha = cv_model.bestModel.stages[-1]._java_obj.getElasticNetParam()
reg_param = cv_model.bestModel.stages[-1]._java_obj.getRegParam()

hyperparam = {
    'alpha': [alpha],
    'reg_param': [reg_param]
}

# 하이퍼파라미터 저장: 최적의 하이퍼파라미터(alpha, reg_param)를 hyperparameter.csv 파일로 저장하여 나중에 사용할 수 있게 한다.
hyper_df = pd.DataFrame(hyperparam).to_csv(f"{data_dir}hyperparameter.csv")
print(hyper_df)
