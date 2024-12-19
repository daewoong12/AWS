from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

# MAX_MEMORY를 설정하여 메모리 크기를 지정하고, SparkSession을 생성
MAX_MEMORY = "5g"
spark = SparkSession.builder.appName("taxi-fare-prediciton")\
                    .config("spark.executor.memory", MAX_MEMORY)\
                    .config("spark.driver.memory", MAX_MEMORY)\
                    .getOrCreate()

# CSV 파일 읽기
trips_df = spark.read.csv("/home/lab01/src/data/trip/*", inferSchema=True, header=True)

# TempView 생성
trips_df.createOrReplaceTempView("trips")

# 스키마 정의
# SQL 쿼리를 사용하여 필요한 컬럼을 추출하고, total_amount, trip_distance, passenger_count 등의 조건을 만족하는 데이터를 필터링
schema = StructType([
    StructField("passenger_count", IntegerType(), True),
    StructField("pickup_location_id", IntegerType(), True),
    StructField("dropoff_location_id", IntegerType(), True),
    StructField("trip_distance", FloatType(), True),
    StructField("pickup_time", IntegerType(), True),
    StructField("day_of_week", StringType(), True),
    StructField("total_amount", FloatType(), True)
])

# SQL 쿼리
query = """
SELECT 
    passenger_count,
    PULocationID as pickup_location_id,
    DOLocationID as dropoff_location_id,
    trip_distance,
    HOUR(tpep_pickup_datetime) as pickup_time,
    DATE_FORMAT(TO_DATE(tpep_pickup_datetime), 'EEEE') AS day_of_week,
    total_amount
FROM
    trips
WHERE
    total_amount < 5000
    AND total_amount > 0
    AND trip_distance > 0
    AND trip_distance < 500
    AND passenger_count < 5
    AND TO_DATE(tpep_pickup_datetime) >= '2021-01-01'
    AND TO_DATE(tpep_pickup_datetime) < '2021-08-01'
"""
data_df = spark.sql(query)  # 데이터 전처리

# 데이터를 train/test로 나누기
train_df, test_df = data_df.randomSplit([0.8, 0.2], seed=42)

# 데이터를 Parquet 형식으로 저장
data_dir = "/home/lab01/src/project/taxi/"
train_df.write.format("parquet").mode('overwrite').save(f"{data_dir}/train/")
test_df.write.format("parquet").mode('overwrite').save(f"{data_dir}/test/")
