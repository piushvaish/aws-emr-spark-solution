# Discuss the purpose of this database in context of the startup, Sparkify, and their analytical goals.

Sparkify want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app. They want an ETL pipeline that extracts their data from S3, processes them using Spark, and loads the data back into S3 as a set of dimensional tables. This will allow their analytics team to continue finding insights in what songs their users are listening to.

# State and justify your database schema design and ETL pipeline.

This project applies Spark and data lakes to build an ETL pipeline for a data lake hosted on S3. Data is loaded from S3 followed by processing the data into analytics tables using Spark. The tables are load back into S3 as parquet files. Spark process is deployed on a EMR cluster using AWS.

## The project uses imperative programming

### Data Descriptions
There are two datasets which are as follows:
* Song Dataset : A subset of real data from the [Million Song Dataset](http://millionsongdataset.com/). Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID.
* Log Dataset : consists of log files in JSON format generated by an [event simulator](https://github.com/Interana/eventsim) and is based on the Song Dataset. These simulate app activity logs from an imaginary music streaming app based on configuration settings. The log files in the dataset are partitioned by year and month. 

### Data pipeline description

The raw datasets are stored in AWS s3 buckets in us-west-2 region. These are processed using a python script and cloud-native big data platform called [Elastic Map Reduce](https://aws.amazon.com/emr/). EMR allows single-purpose, short lived clusters in stand alone mode. The tools include HDFS and Apache Spark. The processed data is loaded back into another s3 bucket as parquet files.

### Description for your model (schema) and included tables

The schema of the the project is star schema. It is optimized for queries on song play analysis. The following tables with the column names.

#### Fact Table
* songplays - records in log data associated with song plays i.e. records with page NextSong
** songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent

#### Dimension Tables
* users - users in the app
** user_id, first_name, last_name, gender, level
* songs - songs in music database
** song_id, title, artist_id, year, duration
* artists - artists in music database
** artist_id, name, location, lattitude, longitude
* time - timestamps of records in songplays broken down into specific units
** start_time, hour, day, week, month, year, weekday

#### Run the scripts

* Enter AWS credentials into the configuration file called dl.cfg
* Change the names of buckets in etl.py to valid names
* Open a python console and enter "run etl.py" to allow the script to read data from S3, process that data using Spark, and write them back to S3


### References:
* https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf
* http://discuss.itversity.com/t/need-help-with-date-format-urgent/13309
* https://www.tutorialspoint.com/pyspark/pyspark_sparkcontext.htm
* https://spark.apache.org/docs/latest/api/python/pyspark.sql.html
* https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/