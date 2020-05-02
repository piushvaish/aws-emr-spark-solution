import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql import functions as F
from pyspark.sql.functions import from_unixtime
from pyspark.sql.functions import monotonically_increasing_id


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """Create Apache Spark session to process the data.
    
    Args:
        None
        
    Returns:
        spark -- Apache Spark session    
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Process Song dataset.
    
    Loads the JSON files, processes the data before selecting the relevant columns.
    The tables are written as parquet files to s3 bucket.
    
    Args:
        spark() : Apache spark session
        input_data(str) : Path for s3 bucket
        output_data(str) : Path for s3 bucket
        
    Returns:
        songs_table -- written to s3 bucket as parquet file
        artists_table -- written to s3 bucket as parquet file    
    """
    # get filepath to song data file
    song_data = input_data + "data/song-data/*/*/*/*.json"
    # "s3://aws-emr-resources-******-us-west-2/notebooks/data/song-data/*/*/*/*/*.json"
    
    # read song data file
    song_df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = song_df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).dropDuplicates()
 
    
    # write songs table to parquet files partitioned by year and artist
    song_file_name = "songs_table.parquet"
    song_file_output = output_data + song_file_name
    songs_table.write.mode("overwrite").partitionBy("year", "artist_id")\
        .parquet(song_file_output)

    # extract columns to create artists table
    artists_table = song_df.select(['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude'])\
                .withColumnRenamed('artist_name','name') \
                .withColumnRenamed('artist_location','location') \
                .withColumnRenamed('artist_latitude','latitude') \
                .withColumnRenamed('artist_longitude','longitude').dropDuplicates()
    
    # write artists table to parquet files
    artists_file_name = "artists_table.parquet"
    artists_file_output = output_data + artists_file_name
    artists_table.write.mode("overwrite").parquet(artists_file_output)
    
    return songs_table, artists_table    


def process_log_data(spark, input_data, output_data):
    """Process Log dataset.
    
    Loads the JSON files, processes the data before selecting the relevant columns.
    The tables are written as parquet files to s3 bucket.
    
    Args:
        spark() : Apache spark session
        input_data(str) : Path for s3 bucket
        output_data(str) : Path for s3 bucket
        
    Returns:
        users_table -- written to s3 bucket as parquet file
        time_table -- written to s3 bucket as parquet file
        songplays_table -- written to s3 bucket as parquet file
    """
    # get filepath to log data file
    log_data = input_data + "data/log-data/*.json"
    # "s3://aws-emr-resources-874610315471-us-west-2/notebooks/data/log-data/*.json"

    # read log data file
    log_df = spark.read.json(log_data)
    
    # filter by actions for song plays
    log_df = log_df.filter(log_df.page == 'NextSong') 

    # extract columns for users table    
    users_table = log_df.select(['userId', 'firstName', 'lastName', 'gender', 'level'])\
            .withColumnRenamed('userId', 'user_id')\
            .withColumnRenamed('firstName', 'first_name')\
            .withColumnRenamed('lastName', 'last_name').dropDuplicates()
    
    # write users table to parquet files
    users_file_name = "users_table.parquet"
    users_file_output = output_data + users_file_name
    users_table.write.mode("overwrite").parquet(users_file_output)

    # create timestamp column from original timestamp column
    log_df = log_df.withColumn("datetime_ts", from_unixtime(F.col('ts')/1000))     
     
    
    # extract columns to create time table
    time_table = log_df.select(['datetime_ts'])\
                    .withColumnRenamed('datetime_ts','start_time') 

    time_table = time_table.withColumn('day', F.dayofmonth('start_time')) \
                      .withColumn('month', F.month('start_time')) \
                      .withColumn('year', F.year('start_time')) \
                      .withColumn('hour', F.hour('start_time')) \
                      .withColumn('minute', F.minute('start_time')) \
                      .withColumn('second', F.second('start_time')) \
                      .withColumn('week', F.weekofyear('start_time')) \
                      .withColumn('weekday', F.dayofweek('start_time')).dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_file_name = "time_table.parquet"
    time_file_output = output_data + time_file_name
    time_table.write.mode("overwrite").partitionBy("year", "month")\
            .parquet(time_file_output)

    # read in song data to use for songplays table
    song_data = input_data + "data/song-data/*/*/*/*.json"
    song_df = spark.read.json(song_data)
    
    # join the two tables
    songplays_table = song_df.join(log_df , (log_df.artist == song_df.artist_name) & \
                     (log_df.song == song_df.title))
    
    # create songplay_id
    songplays_table = songplays_table.withColumn("songplay_id", \
                        monotonically_increasing_id())
    # create the month column in the new table for partitioning
    songplays_table = songplays_table.withColumn('month', F.month('datetime_ts'))

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = songplays_table.select(['songplay_id','year','month', 'ts', 'userId', 'level', 'song_id','artist_id','sessionId','location', 'userAgent'])\
            .withColumnRenamed('ts', 'start_time')\
            .withColumnRenamed('userId', 'user_id')\
            .withColumnRenamed('sessionId', 'session_id')\
            .withColumnRenamed('userAgent', 'user_agent').dropDuplicates()

    # write songplays table to parquet files partitioned by year and month
    songplays_file_name = "songplays_table.parquet"
    songplays_file_output = output_data + songplays_file_name
    songplays_table.write.mode("overwrite").partitionBy("year", "month")\
            .parquet(songplays_file_output)
    
    return users_table, time_table, songplays_table 


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3://aws-emr-resources-******-us-west-2/notebooks/data/processed_data/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
