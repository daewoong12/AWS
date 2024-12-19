from pyspark.sql import SparkSession

MAX_MEMORY="5g"
spark = SparkSession.builder.appName("taxi-fare-prediciton")\
                .config("spark.executor.memory", MAX_MEMORY)\
                .config("spark.driver.memory", MAX_MEMORY)\
                .getOrCreate()

trips_df = spark.read.csv("/home/lab01/src/data/trip/*", inferSchema=True, header=True) # 데이터 프레임 생성

trips_df.createOrReplaceTempView("trips")
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
data_df = spark.sql(query) # 데이터 전처리

train_df, test_df = data_df.randomSplit([0.8, 0.2], seed=1) # 데이터 스플릿
data_dir = "/home/lab01/src/project/taxi/"
train_df.write.format("parquet").mode('overwrite').save(f"{data_dir}/train/")
test_df.write.format("parquet").mode('overwrite').save(f"{data_dir}/test/") # 파이프라인이 여러번 돌 것이기 때문에 overwrite 모드로 지정한다.
