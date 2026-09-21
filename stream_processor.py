# ==============================================================================
# WEEK 4 - CELL 3 : REAL-TIME STREAM PROCESSING WITH PYSPARK
# ==============================================================================
"""
### EXPLANATION FOR NON-TECHNICAL TEAMMATES

1. What is PySpark and why do we need it?
Standard Python (like the Flask API we just built) processes data one piece at a time. 
If a massive bank receives 50,000 credit card swipes in one second, standard Python 
would crash. PySpark is a big data engine that splits the workload across multiple 
"workers" (like hiring an entire team of chefs instead of just one) to process 
massive streams of data without slowing down.

2. What is "Streaming"?
Unlike downloading a static Excel file, streaming means data is flowing continuously, 
like water from a tap. PySpark listens to a specific folder or system and automatically 
processes new data the exact millisecond it arrives.

3. How this script works:
- It creates a "Spark Session" (the master controller).
- It sets up a "Watch Folder" named 'streaming_input'.
- Every time a new transaction file drops into that folder, PySpark immediately picks 
  it up, structures it into columns, and gets it ready for fraud evaluation.

### CODE EXECUTION STEPS
"""

# Step 1: Import necessary PySpark libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.sql.functions import current_timestamp
from pyspark.ml.feature import VectorAssembler

# Step 2: Create the Spark Session
# 'local[*]' tells PySpark to use all available CPU cores on your machine for maximum speed.
print("Initializing PySpark Distributed Streaming Engine...")
spark = SparkSession.builder \
    .appName("FraudDetectionStreamingSystem") \
    .master("local[*]") \
    .getOrCreate()

# We turn off excessive warning messages so our console stays clean and readable
spark.sparkContext.setLogLevel("ERROR")

# Step 3: Define the Data Structure (Schema)
# PySpark needs to know exactly what the incoming data looks like so it can process it instantly.
# This must match the columns in your transaction data.
transaction_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("time_step", DoubleType(), True),
    StructField("v1_feature", DoubleType(), True),
    StructField("v2_feature", DoubleType(), True),
    StructField("v3_feature", DoubleType(), True)
])

# Step 4: Create the Streaming Input Pipeline
# readStream tells PySpark to keep running forever, watching for new files.
# maxFilesPerTrigger=1 means it will process one file at a time to simulate a real-time feed.
# We are telling it to watch a folder called 'streaming_input/'.
streaming_dataframe = spark.readStream \
    .schema(transaction_schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("streaming_input/")

# Step 5: Data Transformation (Feature Engineering)
# We add a timestamp to see exactly when the system processed the transaction.
transformed_stream = streaming_dataframe.withColumn("processed_at", current_timestamp())

# Machine learning algorithms require all numerical features to be combined into a single "Vector".
# VectorAssembler acts as a packer, taking multiple columns and packing them into one called 'features'.
feature_cols = ["amount", "time_step", "v1_feature", "v2_feature", "v3_feature"]
assembler = VectorAssembler(inputCols=feature_cols, outputCols="features")

final_stream = assembler.transform(transformed_stream)

# Step 6: Start the Stream and Output the Results
# writeStream starts the actual engine.
# format("console") tells PySpark to print the processed transactions directly to the screen.
print("PySpark Stream is active. Watching folder 'streaming_input/' for new data...")

stream_query = final_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime="5 seconds") \
    .start()

# This command keeps the script running indefinitely until you forcefully stop it (Ctrl+C).
stream_query.awaitTermination()
