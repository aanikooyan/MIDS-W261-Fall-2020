%pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, TimestampType, Row
from pyspark.sql import SQLContext
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from math import sin, cos, atan2, radians, sqrt
from us import states
import numpy as np

sqlContext = SQLContext(sc)

S3_BUCKET  = "s3://filetransfers3"

airlines = spark.read.option("header", "true").parquet(f"{S3_BUCKET}/airlines/201*.parquet")
weather = spark.read.option("header", "true").parquet(f"{S3_BUCKET}/weather/*.parquet")

open_airport_read_location = f"{S3_BUCKET}/openflights_airports.csv"
open_airport_columns = ["ID", "Name", "City" , "Country", "IATA", "ICAO", 
                        "Latitude", "Longitude", "Altitude", "Timezone", "DST", "zone", "Type", "Source" ]

cleaned_airline_path = f"{S3_BUCKET}/cleaned/airlines"
cleaned_weather_path = f"{S3_BUCKET}/cleaned/weather"
closest_station_path = f"{S3_BUCKET}/cleaned/closest_station_look_up"
airlines_weather_with_dup_path = f"{S3_BUCKET}/cleaned/airline_weather_dup"
airline_weather_path = f"{S3_BUCKET}/cleaned/airline_weather_cleaned"
imputed_data_path = f"{S3_BUCKET}/cleaned/airline_weather_imputed_indicator_added"
# Airlines Data Processing
%pyspark

# assign a unique index to airlines dataset for tracking and joinig later
airlines_idx = airlines.withColumn("FLIGHT_IDX", f.monotonically_increasing_id())

# remove the cancelled flights and keep only the columns of potential interest
keep_columns = ['FLIGHT_IDX', 'YEAR','QUARTER','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 
                'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST', 'DEST_CITY_NAME', 
                'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW','DEP_DEL15', 'DEP_TIME_BLK', 'CRS_ELAPSED_TIME', 
                'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']

not_cancelled = airlines_idx.filter("CANCELLED==0 AND DEP_TIME is not null AND ARR_TIME is not null").select(*keep_columns)

# convert departure and arrival times to timestamp
# actual departure/arrival can happen at midnight, noted as 2400, they are converted to 0:00 of the next day
def extract_datetimestamp(year,month,day,hhmm):
  hour = int(hhmm)//100
  minutes = int(hhmm) - (hour * 100)
  if hour == 24:
      hour = 0
  dt = datetime(int(year),int(month),int(day), hour, minutes)
  if hour == 24:
      dt = dt + timedelta(hours = 24)
  return dt

extract_datetimestamp_udf = f.udf(extract_datetimestamp, TimestampType())
not_cancelled = not_cancelled.withColumn("CRS_DEP_LOCAL", extract_datetimestamp_udf("YEAR","MONTH","DAY_OF_MONTH","CRS_DEP_TIME"))
not_cancelled = not_cancelled.withColumn("DEP_LOCAL", extract_datetimestamp_udf("YEAR","MONTH","DAY_OF_MONTH","DEP_TIME"))
not_cancelled = not_cancelled.withColumn("CRS_ARR_LOCAL", extract_datetimestamp_udf("YEAR","MONTH","DAY_OF_MONTH","CRS_ARR_TIME"))
not_cancelled = not_cancelled.withColumn("ARR_LOCAL", extract_datetimestamp_udf("YEAR","MONTH","DAY_OF_MONTH","ARR_TIME"))

# create a new column that adds two hours to each flight's CRS departure time, 
# when calculating hourly flight rate, this column is used to join back to main flight data
# so that the delay info is joined to flights 2 hours later (effectively giving each flight delay data
# 2 hours before departure.)
delay_add_2h_udf = f.udf(lambda x: x + timedelta(hours = 2), TimestampType())
not_cancelled = not_cancelled.withColumn("CRS_DEP_PLUS2", delay_add_2h_udf("CRS_DEP_LOCAL"))

# create a string yyyy-mm-dd hh, group by this later to calculate hourly flight delay
datetime_hour_udf = f.udf(lambda x: x.strftime("%Y-%m-%d %H"), StringType())
not_cancelled = not_cancelled.withColumn("CRS_DEP_STR", datetime_hour_udf("CRS_DEP_LOCAL"))
not_cancelled = not_cancelled.withColumn("CRS_DEP_PLUS2_STR", datetime_hour_udf("CRS_DEP_PLUS2"))

# rank occurred flights for each tail number in order to joining of previous flight. 
# Rank is sorting by based on departure and arrival time, CRS and actual, because there are errorenous 
# data where two flights of the same physical plane have the same departure/arrival time
not_cancelled = not_cancelled.withColumn("TIME_RANK", f.dense_rank()\
                .over(Window.partitionBy("TAIL_NUM")\
                .orderBy(f.desc("CRS_DEP_LOCAL"), f.desc("DEP_LOCAL"), \
                 f.desc("CRS_ARR_LOCAL"), f.desc("ARR_LOCAL"))))

not_cancelled.cache()

# create a previous flight dataframe, reduce the time rank by 1, so when joining by time rank, 
# the main table's flight is joined to one flight earlier on the previous flight table
prev_flights = not_cancelled.selectExpr('FLIGHT_IDX as PREV_FLIGHT_IDX', 
                                        'TAIL_NUM as PREV_TAIL_NUM',
                                        'TIME_RANK as PREV_TIME_RANK', 
                                        'CRS_DEP_LOCAL as PREV_CRS_DEP_LOCAL', 
                                        'DEP_LOCAL AS PREV_DEP_LOCAL',
                                        'DEP_DELAY as PREV_DEP_DELAY', 
                                        'DEP_DELAY_NEW as PREV_DEP_DELAY_NEW', 
                                        'DEP_DEL15 as PREV_DEP_DEL15', 
                                        'CRS_ELAPSED_TIME as PREV_CRS_ELAPSED_TIME', 
                                        'CRS_ARR_LOCAL as PREV_CRS_ARR_LOCAL', 
                                        'ARR_LOCAL as PREV_ARR_LOCAL',
                                        'ARR_DELAY as PREV_ARR_DELAY', 
                                        'ARR_DELAY_NEW as PREV_ARR_DELAY_NEW', 
                                        'ARR_DEL15 as PREV_ARR_DEL15')

prev_flights = prev_flights.withColumn('PREV_TIME_RANK', prev_flights.PREV_TIME_RANK - 1)

# join main flight and previous flight on time rank and tail number,
# calculate the time from previous flight's actual departure time to the current flight's CRS departure time
# because ACTUAL_DEP_DT_PREV - CRS_DEP_DT will inherently be large for longer flight routes and smaller for short commuter flights
# also subtract the CRS elapsed time of the previous flight
# the result can be conceptually thought of as the time ground crew have to prepare the plane for current flight departure
# short ground crew time and negative ground crew time will likely lead to aircraft delays
joined_prev = not_cancelled.join(prev_flights, (not_cancelled.TAIL_NUM == prev_flights.PREV_TAIL_NUM) & \
                                 (not_cancelled.TIME_RANK == prev_flights.PREV_TIME_RANK))

time_between_udf =  f.udf(lambda x, y: int((x - y).seconds/60), IntegerType())                                
joined_prev = joined_prev.withColumn("PREV_DEP_TO_CURR_CRSDEP", time_between_udf('CRS_DEP_LOCAL', 'PREV_DEP_LOCAL'))
joined_prev = joined_prev.withColumn('PREP_TIME', joined_prev.PREV_DEP_TO_CURR_CRSDEP - joined_prev.PREV_CRS_ELAPSED_TIME)

# number of flight per hour at each airport
hourly_flight = not_cancelled.groupby(['ORIGIN', 'CRS_DEP_PLUS2_STR']).count()
hourly_flight = hourly_flight.selectExpr('ORIGIN as ORIGIN_MERGE', 'CRS_DEP_PLUS2_STR as HOUR_MERGE', 'count as HOURLY_NUM_OF_FLIGHTS')

# number of delayed flights per hour at each airport
hourly_delay = not_cancelled.groupby(['ORIGIN', 'CRS_DEP_PLUS2_STR'])\
                            .agg(f.sum("DEP_DEL15").alias("HOURLY_NUM_OF_DELAY"),\
                                 f.avg("DEP_DELAY").alias("HOURLY_MINUTE_OF_DELAY"), \
                                 f.avg("DEP_DELAY_NEW").alias("HOURLY_MINUTE_OF_DELAY_NEW"))

hourly_flight_merged = hourly_flight.join(hourly_delay, \
                      (hourly_flight.ORIGIN_MERGE == hourly_delay.ORIGIN) & 
                      (hourly_flight.HOUR_MERGE == hourly_delay.CRS_DEP_PLUS2_STR))\
                      .drop(hourly_delay.ORIGIN)\
                      .drop(hourly_delay.CRS_DEP_PLUS2_STR)

# calculate delay rate
hourly_flight_merged = hourly_flight_merged.withColumn("HOURLY_DELAY_RATE",  (f.col("HOURLY_NUM_OF_DELAY") / f.col("HOURLY_NUM_OF_FLIGHTS")))

# join delay info to main flight table
joined_prev_delay = joined_prev.join(hourly_flight_merged, (joined_prev.ORIGIN == hourly_flight_merged.ORIGIN_MERGE) & \
                                     (joined_prev.CRS_DEP_STR == hourly_flight_merged.HOUR_MERGE), 'leftouter')
                                     
open_airports = spark.read.format("csv").option("header","false").load(open_airport_read_location).toDF(*open_airport_columns)
open_airports = open_airports.select("IATA", "Latitude", "Longitude", "Altitude")

airlines_longlat = joined_prev_delay.join(open_airports, joined_prev_delay.ORIGIN == open_airports.IATA)
airlines_longlat = airlines_longlat.withColumn("AIRPORT_LATITUDE", airlines_longlat["Latitude"].cast("double")).drop('Latitude')
airlines_longlat = airlines_longlat.withColumn("AIRPORT_LONGITUDE", airlines_longlat["Longitude"].cast("double")).drop('Longitude')
airlines_longlat = airlines_longlat.withColumn("AIRPORT_ALTITUDE", airlines_longlat["Altitude"].cast("double")).drop('Altitude')

# create a timezone dataframe for the main 50 states + Puerto Rico and Virgin Islands
distinct_states = [x.ORIGIN_STATE_ABR for x in airlines_longlat.select('ORIGIN_STATE_ABR').distinct().collect()]
state_tz_tuple = [(x, states.lookup(x).time_zones[0]) for x in distinct_states if states.lookup(x)]
state_schema = ['state', 'state_tz']
state_timezone = spark.createDataFrame(data = state_tz_tuple, schema = state_schema)

# manually define dataframe for trusted terroritories
tt_tuples = [("Guam, TT",  'Pacific/Guam'), 
             ("Saipan, TT",  'Pacific/Saipan'), 
             ("Pago Pago, TT",  'Pacific/Pago_Pago') ]
tt_schema = ["tt_city",  'tt_tz']
tt_timezone = spark.createDataFrame(data=tt_tuples, schema = tt_schema)

airlines_tz = airlines_longlat.join(tt_timezone, airlines_longlat.ORIGIN_CITY_NAME == tt_timezone.tt_city, 'leftouter')
airlines_tz = airlines_tz.join(state_timezone, airlines_tz.ORIGIN_STATE_ABR == state_timezone.state, 'leftouter')
airlines_tz = airlines_tz.withColumn('ORIGIN_TIMEZONE', f.coalesce('tt_tz', 'state_tz'))

# convert local departure time to UTC using timezone
airlines_tz = airlines_tz.withColumn("CRS_DEP_UTC", f.to_utc_timestamp(f.col("CRS_DEP_LOCAL"), f.col("ORIGIN_TIMEZONE")))

keep_columns = ['FLIGHT_IDX', 'YEAR','QUARTER','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 
                'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_TIMEZONE', 'DEST', 'DEST_CITY_NAME', 
                'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW','DEP_DEL15', 'DEP_TIME_BLK', 'CRS_ELAPSED_TIME', 
                'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 
                'CRS_DEP_LOCAL', 'CRS_DEP_UTC', 'PREV_DEP_TO_CURR_CRSDEP', 'PREP_TIME', 
                'PREV_FLIGHT_IDX', 'PREV_CRS_DEP_LOCAL', 'PREV_DEP_LOCAL', 'PREV_CRS_ELAPSED_TIME', 
                'PREV_DEP_DELAY', 'PREV_DEP_DELAY_NEW', 'PREV_DEP_DEL15', 
                'PREV_CRS_ARR_LOCAL', 'PREV_ARR_LOCAL', 'PREV_ARR_DELAY','PREV_ARR_DELAY_NEW', 'PREV_ARR_DEL15',
                'HOURLY_NUM_OF_FLIGHTS', 'HOURLY_NUM_OF_DELAY', 'HOURLY_DELAY_RATE', 'HOURLY_MINUTE_OF_DELAY', 'HOURLY_MINUTE_OF_DELAY_NEW',
                'AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE', 'AIRPORT_ALTITUDE']

# select cleaned columns and save
airlines_clean = airlines_tz.select(*keep_columns)
airlines_clean.write.parquet(cleaned_airline_path)
# Weather Data Processing
%pyspark
#use udf to filter out only weather data in us
def usa_filter(line):
  parsed_line = line.split(",")
  if len(parsed_line) <2: 
    return False
  _, location = parsed_line
  location_list = location.strip().split(" ")
  if len(location_list) == 2: #example AK US
    country = location_list[1] 
    if country in ["US", "GQ", "AQ"]:
      return True
    else:
      return False
  else:
    return False

usa_filter_udf = f.udf(usa_filter, BooleanType())
us_weather = weather.filter(usa_filter_udf("Name"))

# extract state name for weather stations
schema = StructType([StructField("State", StringType(), False), StructField("StationName", StringType(), False)])

def extract_name_state(line):
  name, remaining = line.split(",")
  state, _ = remaining.strip().split(" ")
  return Row("State", "StationName")(state, name)

extract_name_state_udf = f.udf(extract_name_state, schema)

us_weather = us_weather.withColumn("Output", f.explode(f.array(extract_name_state_udf(us_weather["Name"]))))
us_weather = us_weather.select("*", "Output.*").drop(us_weather["Name"])

# udf will return null values for any readings with bad qualities
bad_quality = ['2', '3', '6', '7']

# Wind Speed, Wind Angle
schema = StructType([StructField("WindSpeed", IntegerType(), False),
                     StructField("WindAngle", IntegerType(), False)])

def extract_wind(line):
    angle, directionQuality, types, speed, speedQuality = line.split(",")
    if directionQuality in bad_quality:
        angle = 999
    if speedQuality in bad_quality:
        speed = 9999
    return Row("WindSpeed", "WindAngle")(int(speed), int(angle))

extract_wind_udf = f.udf(extract_wind, schema)

preprocess_weather = us_weather.withColumn("Output", f.explode(f.array(extract_wind_udf(us_weather["WND"]))))
preprocess_weather = preprocess_weather.select("*", "Output.*").drop(us_weather["WND"]).drop("Output")

# Vertical Visibility
def extract_cig(line):
  verticalVisibility, quality, determination, cavok = line.split(",")
  if quality in bad_quality:
      verticalVisibility = 99999
  return int(verticalVisibility)

extract_cig_udf = f.udf(extract_cig, IntegerType())
preprocess_weather = preprocess_weather.withColumn("VerticalVisibility", extract_cig_udf(preprocess_weather["CIG"])).drop("CIG")

# Visibility
def extract_vis(line):
  visibility, quality, variability, variability_quality = line.split(",")
  if quality in bad_quality:
      visibility = 99999
  return int(visibility)

extract_vis_udf = f.udf(extract_vis, IntegerType())

preprocess_weather = preprocess_weather.withColumn("Visibility", extract_vis_udf(preprocess_weather["VIS"])).drop("VIS")

# Temperature
def extract_tmp(line):
  tmp, quality = line.split(",")
  if quality in bad_quality:
      tmp = 9999
  return float(tmp)/10

extract_tmp_udf = f.udf(extract_tmp, DoubleType())

preprocess_weather = preprocess_weather.withColumn("Temperature", extract_tmp_udf(preprocess_weather["TMP"])).drop("TMP")

# Dew point temperature
def extract_dew(line):
  dtmp, quality = line.split(",")
  if quality in bad_quality:
      dtmp = 9999
  return float(dtmp)/10

extract_dew_udf = f.udf(extract_dew, DoubleType())

preprocess_weather = preprocess_weather.withColumn("DewPointTemp", extract_dew_udf(preprocess_weather["DEW"])).drop("DEW")

# Sea Level Pressure
def extract_slp(line):
  slp, quality = line.split(",")
  if quality in bad_quality:
      slp = 99999
  return int(slp)

extract_slp_udf = f.udf(extract_slp, IntegerType())

preprocess_weather = preprocess_weather.withColumn("SeaLevelPressure", extract_slp_udf(preprocess_weather["SLP"])).drop("SLP")

schema2 = StructType([StructField("CloudMode", IntegerType(), False), StructField("CloudHeight", IntegerType(), False)])

# Cloud Height, Cloud Mode
def extract_cloud(line):
    if line == '':
        return Row("CloudMode", "CloudHeight")(int(9), int(99999))
    else:
        code, oktas, cquality, height, dquality, character  = line.split(",")
        if cquality in bad_quality:
            code = 9
        if dquality in bad_quality:
            height = 99999
        return Row("CloudMode", "CloudHeight")(int(code), int(height))

extract_cloud_udf = f.udf(extract_cloud, schema2)

preprocess_weather = preprocess_weather.withColumn("Output", f.explode(f.array(extract_cloud_udf(preprocess_weather["GD1"]))))
preprocess_weather = preprocess_weather.select("*", "Output.*").drop(us_weather["GD1"]).drop("Output")

# Weather Condition
def extract_aw(line):
    if line == '':
        return int(9999)
    code, quality = line.split(",")
    if quality in bad_quality:
        code = 9999
    return int(code)

extract_aw_udf = f.udf(extract_aw, IntegerType())

preprocess_weather = preprocess_weather.withColumn('AW', f.coalesce('AW1', 'AW2', 'AW3', 'AW4', 'AW5', 'AW6', 'AW7')).drop('AW1', 'AW2', 'AW3', 'AW4', 'AW5', 'AW6', 'AW7')
preprocess_weather = preprocess_weather.withColumn("AtmCondition", extract_aw_udf(preprocess_weather["AW"])).drop("AW")

# remove the null value placeholders
def find_null(value, null_placeholder):
    return f.when(value != null_placeholder, value).otherwise(f.lit(None))

null_pairs = [('WindAngle', 999), ('WindSpeed', 9999), ('VerticalVisibility', 99999),
              ('Visibility', 999999), ('Temperature', 999.9), ('DewPointTemp', 999.9),
              ('SeaLevelPressure', 99999), ('CloudMode', 9), ('CloudHeight', 99999),
              ('AtmCondition', 9999)]

for pair in null_pairs:
    col_name, null_value = pair
    preprocess_weather = preprocess_weather.withColumn(col_name, find_null(f.col(col_name), null_value))

# save
weather_clean = preprocess_weather.selectExpr(["STATION as WeatherStationID", "StationName as WeatherStationName", 
    "LATITUDE as WeatherStationLatitude", "LONGITUDE as WeatherStationLongitude", "DATE as WeatherTimestamp", 
    'WindAngle', 'WindSpeed', 'VerticalVisibility', 'Visibility', 'Temperature', 
    'DewPointTemp','SeaLevelPressure', 'CloudMode', 'CloudHeight', 'AtmCondition'])

weather_clean.write.parquet(cleaned_weather_path)
# Joining Airlines and Weather
%pyspark

airlines_clean = spark.read.option("header", "true").parquet(cleaned_airline_path)
weather_clean = spark.read.option("header", "true").parquet(cleaned_weather_path)
open_airports = spark.read.format("csv").option("header","false").load(open_airport_read_location).toDF(*open_airport_columns)

# get distinct airports in the airlines dataset, join with iata table for long/lat
unique_iata_code = airlines_clean.select('ORIGIN').distinct()
unique_iata_location = unique_iata_code.join(open_airports, unique_iata_code.ORIGIN == open_airports.IATA)\
                      .select("IATA", "Latitude", "Longitude", "Altitude")

# For every airport, find the closest weather station.
us_weather_station = weather_clean.select("WeatherStationID", "WeatherStationLatitude", "WeatherStationLongitude").distinct()

possible_pairs = unique_iata_location.join(us_weather_station)
possible_pairs_rdd = possible_pairs.rdd

# compute distance between each pair of airport and weather station
def compute_distance(line):
  code, lat1, lon1, lat, station, lat2, lon2 = line
  lat1 = float(lat1)
  lon1 = float(lon1)
  lat2 = float(lat2)
  lon2 = float(lon2)
  
  radius = 6371 # km
  dlat = radians(lat2-lat1)
  dlon = radians(lon2-lon1)
  a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) \
      * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  d = radius * c
  return code, station, d

def get_min(a,b):
  if a[1] < b[1]:
    return a
  else:
    return b

pairs_rdd_with_distance = possible_pairs_rdd.map(compute_distance).cache()

# select the 3 closest weather station for each airport
iata_three_closest_stations = pairs_rdd_with_distance.toDF(["IATA", "WeatherStationID", "Distance"])\
    .select("*", f.row_number()\
    .over(Window.partitionBy("IATA").orderBy(f.col("Distance"))).alias("rowNum"))\
    .where(f.col("rowNum") <= 3).drop("rowNum")

# remove the station if more than threshold distance away
threshold_km = 30
closest_stations = iata_three_closest_stations.filter(f"Distance < {threshold_km}")

closest_stations.write.parquet(closest_station_path)%pyspark

airlines_clean = spark.read.option("header", "true").parquet(cleaned_airline_path)
weather_clean = spark.read.option("header", "true").parquet(cleaned_weather_path)
closest_stations = spark.read.option("header", "true").parquet(closest_station_path)

airlines_join = airlines_clean.join(closest_stations, 
                                 airlines_clean.ORIGIN == closest_stations.IATA, "inner")\
                                .select(['FLIGHT_IDX', 'WeatherStationID', 'CRS_DEP_UTC'])

# add time range for join (2 ~ 3 hours prior departure time)
start_datetime_udf = f.udf(lambda x: x + timedelta(hours = -3), TimestampType())
end_datetime_udf = f.udf(lambda x: x + timedelta(hours = -2), TimestampType())
airlines_join = airlines_join.withColumn("WEATHER_START", start_datetime_udf("CRS_DEP_UTC"))
airlines_join = airlines_join.withColumn("WEATHER_END", end_datetime_udf("CRS_DEP_UTC"))

# join airlines and weather
airlines_weather = airlines_join.join(weather_clean, (weather_clean.WeatherStationID == airlines_join.WeatherStationID) \
                                      & (weather_clean.WeatherTimestamp > airlines_join.WEATHER_START) \
                                      & (weather_clean.WeatherTimestamp < airlines_join.WEATHER_END ))\
                                      .drop(weather_clean.WeatherStationID)

airlines_weather.write.parquet(airlines_weather_with_dup_path)
# Aggregate multiple weather readings per flight
%pyspark
airlines_weather = spark.read.option("header", "true").parquet(airlines_weather_with_dup_path)
airlines_clean = spark.read.option("header", "true").parquet(cleaned_airline_path)

airlines_weather_numeric = airlines_weather\
      .select(['FLIGHT_IDX', 'WindAngle', 'WindSpeed', 'VerticalVisibility', 'Visibility', 
      'Temperature', 'DewPointTemp', 'SeaLevelPressure', 'CloudHeight']).groupby('FLIGHT_IDX').mean()

# numeric values
airlines_weather_numeric = airlines_weather_numeric\
      .selectExpr('FLIGHT_IDX', 
                  '`avg(WindAngle)` as WindAngle', 
                  '`avg(WindSpeed)` as WindSpeed', 
                  '`avg(VerticalVisibility)` as VerticalVisibility', 
                  '`avg(Visibility)` as Visibility', 
                  '`avg(Temperature)` as Temperature', 
                  '`avg(DewPointTemp)` as DewPointTemp', 
                  '`avg(SeaLevelPressure)` as SeaLevelPressure', 
                  '`avg(CloudHeight)` as CloudHeight')

# categorical values
CloudMode_df = airlines_weather.select(['FLIGHT_IDX', 'CloudMode']).where(f.col("CloudMode").isNotNull())
grouped1 = CloudMode_df.groupBy('FLIGHT_IDX', 'CloudMode').count()
window = Window.partitionBy("FLIGHT_IDX").orderBy(f.desc("count"))
CloudMode_mode = grouped1.withColumn('order', f.row_number().over(window)).where(f.col('order') == 1)
CloudMode_mode = CloudMode_mode.select('FLIGHT_IDX', 'CloudMode')

AtmCondition_df = airlines_weather.select(['FLIGHT_IDX', 'AtmCondition']).where(f.col("AtmCondition").isNotNull())
grouped2 = AtmCondition_df.groupBy('FLIGHT_IDX', 'AtmCondition').count()
AtmCondition_mode = grouped2.withColumn('order', f.row_number().over(window)).where(f.col('order') == 1)
AtmCondition_mode = AtmCondition_mode.select('FLIGHT_IDX', 'AtmCondition')

airline_weather_no_dup = airlines_clean.join(airlines_weather_numeric, \
     airlines_clean.FLIGHT_IDX == airlines_weather_numeric.FLIGHT_IDX, 'leftouter')\
    .drop(airlines_weather_numeric.FLIGHT_IDX)

airline_weather_no_dup = airline_weather_no_dup.join(CloudMode_mode, \
     airline_weather_no_dup.FLIGHT_IDX == CloudMode_mode.FLIGHT_IDX, 'leftouter')\
    .drop(CloudMode_mode.FLIGHT_IDX)

airline_weather_no_dup = airline_weather_no_dup.join(AtmCondition_mode, \
     airline_weather_no_dup.FLIGHT_IDX == AtmCondition_mode.FLIGHT_IDX, 'leftouter')\
    .drop(AtmCondition_mode.FLIGHT_IDX)

keep_columns = ['FLIGHT_IDX', 'QUARTER', 'MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'CRS_DEP_TIME', 'DEP_TIME_BLK', 
                'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_TIMEZONE',
                'AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE', 'AIRPORT_ALTITUDE', 'DEST', 'DEST_CITY_NAME', 'CRS_ELAPSED_TIME', 
                'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'CRS_DEP_LOCAL', 'CRS_DEP_UTC',
                'PREV_DEP_TO_CURR_CRSDEP', 'PREP_TIME', 'HOURLY_DELAY_RATE', 'HOURLY_NUM_OF_FLIGHTS', 'HOURLY_NUM_OF_DELAY',
                'PREV_FLIGHT_IDX', 'PREV_CRS_DEP_LOCAL', 'PREV_DEP_LOCAL', 'PREV_DEP_DELAY', 'PREV_DEP_DELAY_NEW', 'PREV_DEP_DEL15', 'PREV_CRS_ELAPSED_TIME',
                'PREV_CRS_ARR_LOCAL', 'PREV_ARR_LOCAL', 'PREV_ARR_DELAY', 'PREV_ARR_DELAY_NEW', 'PREV_ARR_DEL15', 
                'WindAngle', 'WindSpeed', 'VerticalVisibility', 'Visibility', 'Temperature', 'DewPointTemp', 'SeaLevelPressure', 'CloudHeight', 'CloudMode', 'AtmCondition']

cleaned_data = airline_weather_no_dup.select(*keep_columns)
cleaned_data.write.parquet(airline_weather_path)
# Impute missing weather data
%pyspark
cleaned_data = spark.read.option("header", "true").parquet(airline_weather_path)

# since weather data is 2-3 hours before CRS_DEP_LOCAL, 
# subtract 2.5 hours as rough timestmap for weather data
def timestamp_to_block(dt):
  weather_timestamp = dt + timedelta(hours = -2.5)
  hour = int(weather_timestamp.strftime('%H'))
  if (hour >= 0) & (hour < 6):
      hour_block = 'early'
  elif (hour >= 6) & (hour < 12):
      hour_block = 'morning'
  elif (hour >= 12) & (hour < 18):
      hour_block = 'afternoon'
  elif (hour >= 18) & (hour < 24):
      hour_block = 'night'
  return f'{weather_timestamp.strftime("%Y-%m-%d")} {hour_block}'

datetime_block_udf = f.udf(timestamp_to_block, StringType())

cleaned_data = cleaned_data.withColumn("Weather_Timeblock", datetime_block_udf("CRS_DEP_UTC"))

# impute
def median(values_list):
    med = np.median(values_list)
    return float(med)

udf_median = f.udf(median, DoubleType())

df_medians = cleaned_data.groupBy('ORIGIN','Weather_Timeblock').\
            agg(udf_median(f.collect_list(f.col('WindSpeed'))).alias('WindSpeed_md'), \
                udf_median(f.collect_list(f.col('VerticalVisibility'))).alias('VerticalVisibility_md'), \
                udf_median(f.collect_list(f.col('Visibility'))).alias('Visibility_md'), \
                udf_median(f.collect_list(f.col('Temperature'))).alias('Temperature_md'), \
                udf_median(f.collect_list(f.col('DewPointTemp'))).alias('DewPointTemp_md'),\
                udf_median(f.collect_list(f.col('CloudHeight'))).alias('CloudHeight_md'))
                
impute_data = cleaned_data.join(df_medians, (cleaned_data.ORIGIN == df_medians.ORIGIN) & (cleaned_data.Weather_Timeblock == df_medians.Weather_Timeblock), 'leftouter').drop(df_medians.ORIGIN)

impute_data = impute_data.withColumn('WindSpeed_imp', f.coalesce('WindSpeed', 'WindSpeed_md'))
impute_data = impute_data.withColumn('VerticalVisibility_imp', f.coalesce('VerticalVisibility', 'VerticalVisibility_md'))
impute_data = impute_data.withColumn('Visibility_imp', f.coalesce('Visibility', 'Visibility_md'))
impute_data = impute_data.withColumn('Temperature_imp', f.coalesce('Temperature', 'Temperature_md'))
impute_data = impute_data.withColumn('DewPointTemp_imp', f.coalesce('DewPointTemp', 'DewPointTemp_md'))
impute_data = impute_data.withColumn('CloudHeight_imp', f.coalesce('CloudHeight', 'CloudHeight_md'))

# new indicator columns from EDA
def cloudmode_binarize(cm):
    if cm == np.nan:
        return 0
    else:
        if cm == 0:
            return 1
        else:
            return 0

cloudmode_binarize_udf = f.udf(cloudmode_binarize, IntegerType())
impute_data = impute_data.withColumn('CloudMode_binary', cloudmode_binarize_udf('CloudMode'))

timeblock6_udf = f.udf(lambda x: 1 if x == '0600-0659' else 0, IntegerType())
timeblock7_udf = f.udf(lambda x: 1 if x == '0700-0759' else 0, IntegerType())
timeblock0_udf = f.udf(lambda x: 1 if x == '0001-0559' else 0, IntegerType())
timeblock8_udf = f.udf(lambda x: 1 if x == '0800-0859' else 0, IntegerType())
impute_data = impute_data.withColumn("BLOCK6", timeblock6_udf("DEP_TIME_BLK"))
impute_data = impute_data.withColumn("BLOCK7", timeblock7_udf("DEP_TIME_BLK"))
impute_data = impute_data.withColumn("BLOCK0", timeblock0_udf("DEP_TIME_BLK"))
impute_data = impute_data.withColumn("BLOCK8", timeblock8_udf("DEP_TIME_BLK"))

impute_data = impute_data.select('FLIGHT_IDX', 'QUARTER', 'MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'CRS_DEP_TIME', 'DEP_TIME_BLK', 'CRS_DEP_UTC', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_TIMEZONE', 'AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE', 'AIRPORT_ALTITUDE', 'DEST', 'DEST_CITY_NAME', 'CRS_ELAPSED_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'PREV_DEP_TO_CURR_CRSDEP', 'PREP_TIME', 'HOURLY_DELAY_RATE', 'HOURLY_NUM_OF_FLIGHTS', 'HOURLY_NUM_OF_DELAY', 'PREV_FLIGHT_IDX', 'PREV_CRS_DEP_LOCAL', 'PREV_DEP_LOCAL', 'PREV_DEP_DELAY', 'PREV_DEP_DELAY_NEW', 'PREV_DEP_DEL15', 'PREV_CRS_ELAPSED_TIME', 'PREV_CRS_ARR_LOCAL', 'PREV_ARR_LOCAL', 'PREV_ARR_DELAY', 'PREV_ARR_DELAY_NEW', 'PREV_ARR_DEL15', 'WindAngle', 'WindSpeed_imp', 'VerticalVisibility_imp', 'Visibility_imp', 'Temperature_imp', 'DewPointTemp_imp', 'SeaLevelPressure', 'CloudHeight_imp', 'CloudMode', 'AtmCondition', 'CloudMode_binary', 'BLOCK6', 'BLOCK7', 'BLOCK0', 'BLOCK8')

delay_cols = ['HOURLY_DELAY_RATE', 'HOURLY_NUM_OF_FLIGHTS']
impute_data = impute_data.na.fill(0, subset = delay_cols)

drop_na_cols = ['ORIGIN', 'FL_DATE', 'MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK', 'AIRPORT_ALTITUDE', 'PREP_TIME', 'HOURLY_DELAY_RATE', 'HOURLY_NUM_OF_FLIGHTS', 'PREV_DEP_DEL15', 'PREV_CRS_ELAPSED_TIME', 'PREV_ARR_DEL15', 'WindSpeed_imp', 'VerticalVisibility_imp', 'Visibility_imp', 'Temperature_imp', 'DewPointTemp_imp', 'DEP_DELAY_NEW', 'DEP_DEL15']

impute_final = impute_data.na.drop(how = 'any', subset = drop_na_cols)
impute_final = impute_final\
    .filter('(PREV_CRS_ELAPSED_TIME > 18) AND (CRS_ELAPSED_TIME > 18)')\
    .filter('(PREV_DEP_DELAY > -45) AND (PREV_DEP_DELAY < 1609)')\
    .filter('(PREV_ARR_DELAY > -81) AND (PREV_ARR_DELAY < 1605)')\
    .filter('(DEP_DELAY > -45) AND (DEP_DELAY < 1609)')

impute_final.write.parquet(imputed_data_path)%pyspark
