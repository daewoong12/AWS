from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
import pandas as pd

MAX_MEMORY = "5g"
spark = SparkSession.builder.appName("taxi-fare-prediciton").config("spark.executor.memory", MAX_MEMORY).config("spark.driver.memory", MAX_MEMORY).getOrCreate()

# 스키마 정의
# schema를 명시적으로 설정하여 데이터가 올바른 형태로 읽히도록 합니다.
schema = StructType([
    StructField("passenger_count", IntegerType(), True),
    StructField("pickup_location_id", IntegerType(), True),
    StructField("dropoff_location_id", IntegerType(), True),
    StructField("trip_distance", FloatType(), True),
    StructField("pickup_time", IntegerType(), True),
    StructField("day_of_week", StringType(), True),
    StructField("total_amount", FloatType(), True)
])

# train_df와 test_df를 Parquet 파일에서 읽어옵니다.
data_dir = "/home/lab01/src/project/taxi/"
train_df = spark.read.parquet(f"{data_dir}/train/")
test_df = spark.read.parquet(f"{data_dir}/test/")

# 하이퍼파라미터 불러오기
# hyperparameter.csv에서 alpha(ElasticNet 파라미터)와 reg_param(정규화 파라미터)를 불러온다
hyper_df = pd.read_csv(f"{data_dir}hyperparameter.csv")
alpha = float(hyper_df.iloc[0]['alpha'])
reg_param = float(hyper_df.iloc[0]['reg_param'])

cat_feats = ["pickup_location_id", "dropoff_location_id", "day_of_week"]
stages = []

# 범주형 변수 처리 -> StringIndexer와 OneHotEncoder를 사용하여 처리
for c in cat_feats:
    cat_indexer = StringIndexer(inputCol=c, outputCol=c + "_idx").setHandleInvalid("keep")
    onehot_encoder = OneHotEncoder(inputCols=[cat_indexer.getOutputCol()], outputCols=[c + "_onehot"])
    stages += [cat_indexer, onehot_encoder]

# 수치형 변수 처리 -> VectorAssembler와 StandardScaler로 변환
num_feats = ["passenger_count", "trip_distance", "pickup_time"]
for n in num_feats:
    num_assembler = VectorAssembler(inputCols=[n], outputCol=n + "_vector")
    num_scaler = StandardScaler(inputCol=num_assembler.getOutputCol(), outputCol=n + "_scaled")
    stages += [num_assembler, num_scaler]

assembler_inputs = [c + "_onehot" for c in cat_feats] + [n + "_scaled" for n in num_feats]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="feature_vector")
stages += [assembler]

# 파이프라인 구성
pipeline = Pipeline(stages=stages)
fitted_transformer = pipeline.fit(train_df)

# 모델 학습
vtrain_df = fitted_transformer.transform(train_df)
lr = LinearRegression(maxIter=50, solver="normal", labelCol="total_amount", featuresCol="feature_vector", elasticNetParam=alpha, regParam=reg_param)
model = lr.fit(vtrain_df)

# 예측
vtest_df = fitted_transformer.transform(test_df)
predictions = model.transform(vtest_df)

# 예측 결과 출력("trip_distance", "day_of_week", "total_amount", "prediction" 출력)
# trip_distance: 실제 여행 거리 값입니다. 모델의 입력 변수로 사용된 값입니다.
# day_of_week: 여행이 발생한 요일을 나타냅니다. 이는 범주형 변수로 처리되어 OneHotEncoder로 변환됩니다.
# total_amount: 실제 택시 요금입니다. 이는 모델이 예측하려는 대상 값입니다.
# prediction: 모델이 예측한 택시 요금입니다. trip_distance와 day_of_week를 기반으로 예측된 값입니다.
predictions.select(["trip_distance", "day_of_week", "total_amount", "prediction"]).show()

# 모델 저장
model.write().overwrite().save("/home/lab01/src/project/taxi/airflow")
