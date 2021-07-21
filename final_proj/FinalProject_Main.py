#!/usr/bin/env python
# coding: utf-8

# # 1. Question Formulation
#  Flight delay is a common occurrence in air travel.  In the United States, approximately 20% flights have a departure delay of greater than 15 minutes. [1] In 2019, the Federal Aviation Administration estimated the annual cost of delays to be 33 billion, factoring in direct cost to airlines and passengers, welfare loss due to lost air travel demand, and indirect cost as lost in productivity. From a business perspective, reducing the cost associated with flight delay would improve efficiency and increase profits for the airlines, and also benefit the passenger by minimizing the time lost due to traveling. [2] Flight delay prediction is important for operating airlines and airport to plan in advance to accommodate significant flight delays and reduce the frequency of delay occurrence. 
# The goal of the present study is to predict the occurrence of flight departure delays of longer than 15 minutes. Departure delay is defined as the elapsed time between the scheduled departure and the actual departure time of a flight.
# 
# ## State-of-the-art research
# Ongoing research on this subject commonly utilize Gradient Boosting Classifiers [3], random forest [4], AdaBoost and k-Nearest Neighbors [5], for flight departure and arrival delays. Chakrabarty [3] used a Grid Search to optimize the hyperparameters for Gradient Boosting Classifier Model and achieved a validation accuracy of 85.73% in a balanced dataset. A Convolutional neural network model created by Jiang et al. [6] predicted multi-class flight delay with 89.32% prediction accuracy.
# 
# The dataset we used were the Flight On-Time Performance Data [1] from U.S. Department of Transportation and Weather station records [7] from National Oceanic and Atmospheric Administration. Past research have shown that weather observations improve flights delay prediction performances. [3][4][5].
# 
# In this study we implement a supervisied machine learning classification algorithm on scale in order to learn from (and predict on) the flight data obtained across all airpirts within the US as well as weather data obtained from weather stations across the country over the 2015-2019 period. To optimize the performance of our machine learning model, we seek to understand the most important contributing factors to flight delays, the inherent limitations associated with the the available data, the type of algorithm best suited for the prediction task, and the volume of data required to train the model.
# 
# Given that flight delay and on-time ratio are highly unbalanced, realistically, the implemented algorithm and metrics used to assess the performance should be selected to address this concern. As it will be discussed in details in the modeling section, our final choice for the algorithm was the ensemble random forest. 
# 
# ## Metric of Choice
# Although Receiver Operating Characteristic (ROC) curves are widely used to evaluate binary classifiers, they may not be the best choice in terms of reliability when the problem of class imbalance is associated to the presence of a low sample size of minority instances, since may provide an excessively optimistic view of the performance. The tradeoff between precision and recall, on the other hand, make it possible to better assess the performance of a classifier on the minority class.
# 
# Precision is defined as the number of correct positive predictions made: TP/(TP + FP)
# 
# Recall quantifies the number of correct positive predictions made out of all positive predictions that could have been made: TP/(TP + FN)
# 
# In the flight delay classification problem, the negative and positive labels would be defined as no delay and delay, respectively. From a business standpoint, both false positives (FP), i.e. predicting no-delay flight as delayed, and false negatives (FN), i.e. predicting a delayed flight as without delay, can be harmful, although not necessarily at similar scales. Thus, the metrics to evaluate the performance of such classifier should not only fit better to the data with severe class imbalance, but also be able to keep a good balance between the number of FP and FN predictions. 
# 
# F1-score is a universal metric that has been used by many ML experts and is a representation of the balance between precision and recall. F1-score is defined as : 2x (precision x recall) / (precision + recall). Thus, it will be used as the main metrics to assess the performance of our classifier. The target goal is to have a minimum F1-score of 0.7. In addition, to compare between different classifiers and also to other studies, we also consider accuracy and area under the ROC curve as a secondary metrics. 
# 
# [1] Flight on-time performance data from TranStats data collection, U.S. Department of Transportation. https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time
# [2] Cost delay estimates 2019. Federal Aviation Agency https://www.faa.gov/data_research/aviation_data_statistics/media/cost_delay_estimates.pdf
# [3] Chakrabarty, N., (2019), “A data mining approach to flight arrival delay prediction for American airlines”, At 9th Annual Information Technology Electromechanical Engineering and Microelectronics Conference, Jaipur, India.
# [4] Belcastro, L., Marozzo, F., Talia, D., & Trunfio, P. (2016). Using Scalable Data Mining for Predicting Flight DelaysACM Trans. Intell. Syst. Technol., 8(1).
# [5] S. Choi, Y. J. Kim, S. Briceno and D. Mavris, "Prediction of weather-induced airline delays based on machine learning algorithms," 2016 IEEE/AIAA 35th Digital Avionics Systems Conference (DASC), Sacramento, CA, 2016, pp. 1-6, doi: 10.1109/DASC.2016.7777956.
# [6] Y. Jiang, Y. Liu, D. Liu and H. Song, "Applying Machine Learning to Aviation Big Data for Flight Delay Prediction," 2020 IEEE Intl Conf on Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech), Calgary, AB, Canada, 2020, pp. 665-672, doi: 10.1109/DASC-PICom-CBDCom-CyberSciTech49142.2020.00114.
# [7] Weather station records, National Oceanic and Atmospheric Administration, https://www.ncdc.noaa.gov/orders/qclcd/
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# 
# # 2. EDA & Discussion of Challenges
# 
# ### The start to finish data cleaning code can be found here: [FinalProject_AuxiliaryNotebook_Data_Cleaning_Imputation](http://44.237.112.83:8890/#/notebook/2FTEHBEG5)
# ### The codes and outputs for EDA can be found here: [FinalProject_AuxiliaryNotebook_EDA](http://44.237.112.83:8890/#/notebook/2FTTAUFYU)
# 
# ## EDA Overview
# 
# The following are the goals of the EDA:
# * Understand the general structure and size of the data
# * Understand the extend of missing data and the impact
# * Discover any outlier values and potential erroneous data
# * Obtain rough distributions of the variables of interest
# * Explore any potential relationship between the candidate predictors and the outcome variable, departure delays of over 15 minutes, `DEP_DEL15`.
# * Remove highly correlated features
# * Select only the most important features to include in the model from all available variables to reduce processing cost and improve scalability
# * Infer possible approaches to feature engineering
# 
# ## Original flight data and exclusions
# 
# The original flight dataset contains 31,746,841 flights. Of those, 489,947 were cancelled flights. Because the intent of this project is to predict departure delays, which does not apply to cancelled flights, the cancelled flights were excluded from the dataset before merging to weather dataset and data cleaning. In the original dataset, delayed flights make up 18.22% of non-cancelled flights. 
# 
# After excluding the cancelled flights, we examined missing values in the key columns. There are no missing values in flight date `FL_DATE`, CRS departure time`CRS_DEP_TIME`, actual departure time `DEP_TIME`, and actual arrival time `CRS_ARR_TIME`. There are 4744 rows missing values for `DEP_DEL15`, the most important columns, which is the outcome variable we intent to predict. It's unknown why these flights are missing this value. A cursory look at the this subset of flight showed some are also missing arrival time `ARR_TIME`, so there could be some special circumstances surrounding these flights not documented in the data. Since we cannot confirm whether these flights actually took off and landed as normal, and that these compose of only 0.02% of the data, these flights were excluded.
# 
# Additionally, we looked for outliers of key flight data columns. Quantile estimation of flights with negative departure delays, showed that vavst majority of the flights that departed ahead of scheduled time were early by less than 20 minutes. Only 0.001% were early by 45 minutes or more. The minimal for departure delay is 234 minutes, almost 4 hours ahead of schedule, which is highly improbable in practice, and more likely to be a data entry error. Therefore, flights with delay times below 0.00001th quantile of negative delay times were excluded. The maximum for departure delay 2755 minutes, almost 2 days, which is also highly improbable. To avoid using unreliable data, flights with delay time above 0.99999th quantile of positive delay times were also excluded. Similar treatments were performed on arrival delays. For CRS_elapsed_time, the 0.00001th quantile is 20 minutes, filtering for flights under 20 minutes showed that there is a flight in Alaska with 18 minutes flight time, and they appear to be accurate. There are fewer than 20 flights with durations shorter than 17 minutes. Almost all appear to be erroreous and should be excluded, as flights cross states cannot possible lands within 17 minutes of take off. As our processed flight data including joining of the previous flight of the same plane, flights joined to these flights with questionable data are also dropped before joining weather data.
# 
# Overall, the exclusions removed less than 2% of the original dataset. The cleaned flight data includes 31,117,414 flights. Of these, 18.07% had departure delay of over 15 minutes, which is a similar proportion to the original data, so the exclusions likely did not affect the general distribution of the dataset.
# 
# ## Flight data EDA
# 
# Univariate analysis shows that about 18% flights are delayed for more than 15 minutes. The amount of delay is defined as time between scheduled departure to actual departure, if the actual departure occured after scheduled time. The mean of delay time is 12 minutes, with a standard deviation of 41 minutes. While the maximum of delay time after exclusion is around 26.7 hours, the quantile approximation earlier and histogram below shows that the vast majority of delay is less 25 minutes.
# 
# ![Figure 1](https://raw.githubusercontent.com/jying-data/imghost/master/1.png)
# 
# We considered several variables directly available from the flight dataset as potential predictors for delay. 
# 
# In analysis of the mean delay duration in a sampled dataset, we note that the mean is highest during in June, July and December, suggesting some seasonal patterns to amount of flight delay. The delay in the summer could be due to summer travels or weather related delays, such as storms commonly occuring the season. The delay in December is likely associated with holiday season travels. The seasonality effect is less obvious with day of of the week, though it seems that Monday, Friday and Sundays tend to have the higher delays, possibly due to vacation related trips being scheduled next to a weekend.
# 
# There is an obvious effect of scheduled departure the amount of delay time. Delay time is shortest at early hours during the day before 6am, and then consistently increases until peaking at 9pm, after which it declines again. This can be attributed to normal airport operations, where the delay time accumulates in a chain reaction as previous flights delay causes the subsequent flight delays. and the delay dissipates towards the end of the night where are fewer flights departing.
# 
# We observe no obvious differences of flight carrier on amount of delay. The carriers with shorter average delay time are also carrier that operate fewer flights, so the more likely explanation is traffic, rather than carrier. 
# 
# ![Figure 2](https://raw.githubusercontent.com/jying-data/imghost/master/2.png)
# 
# 
# Analysis of counts of flights delayed for more than 15 minutes in the full dataset showed similar patterns noted above. Overlaying the rate of delayed flights on the number of flights shows that the delay rates tend to be higher when there are more flights and vice versa.
# ![Figure 3](https://raw.githubusercontent.com/jying-data/imghost/master/3.png)
# ![Figure 4](https://raw.githubusercontent.com/jying-data/imghost/master/4.png)
# ![Figure 5](https://raw.githubusercontent.com/jying-data/imghost/master/5.png)
# 
# ## Processed flight data EDA
# 
# As noted in the earlier section, chain delays happen frequently and tend to build up throughout the delay. Therefore, an intuitive follow-up is to examine the amount of delay time and delay rate in relation to previous flights. During processing of the flight data, we joined each flight to the previous occured flight (i.e. not cancelled) based on plane tail number. The logic is that if a plane arrived late, then the return trip will likely be delayed.
# 
# Boxplot of delay time of sample data set shows this pattern. In flights that were ultimately delayed for more than 15 minutes, the median departure and arrival delay time of the previous flight were both significantly higher. 
# 
# ![Figure 6](https://raw.githubusercontent.com/jying-data/imghost/master/6.png)
# 
# Relating to this, we considered the possible effect of previous flight's duration to account for the common experience of a flight making up for its departure delay in flight. Longer flights would have more opportunity to catch up than shorter flights and might arrive on time despite departure delays. To proxy this effect, we took the differences between previous flight's actual departure time and next flight's CRS departure time, and then subtracted the CRS flight elapsed time. The result of this can be conceptualized as the amount of preparation time the airport ground crew have, between when a plane lands, and when the next flight is scheduled to depart. Overall, the amount of prep time is shorter for flights that ended up delayed.
# 
# ![Figure 7](https://raw.githubusercontent.com/jying-data/imghost/master/7.png)
# 
# Grouping previous flight's on time status with current flight's status show a very large disparity in delay rate between flights whose previous flight was on time and whose previous flight was delayed. The conditional probability of a flight being delayed when the previous flight departed on time is about 12%, while the conditional probability of of a flight being delayed when the previous flight departed more than 15 minutes late jumps up to about 48%. The disparity is greater for when the previous flight arrived late, about 10% and 53% for on time and late arrivals, respectively. However, despite previous flight arrival delay being a great predictor, often times, especially with short commute flights, the previous flight would not arrive more than 2 hours ahead of next flight's scheduled departure, so realistically, we cannot expect this data to be available in field application, and therefore previous flight arrival delay should be omitted from the model.
# 
# 
# ![Figure 8](https://raw.githubusercontent.com/jying-data/imghost/master/8.png)
# 
# Moreover, to investigate the correlation between flight traffic and delay rates, we proxied this effect by looking at the number of flights scheduled to depart the same airport 2 hours before a flight, as well as the number of delayed flights at the airport in the same time window. In general, the flights ended up with departure delays had slightly more flights departing 2 hours before its scheduled departure, and higher delay rate 2 hours before.
# 
# ![Figure 9](https://raw.githubusercontent.com/jying-data/imghost/master/9.png)
# 
# ## Weather data imputation and EDA
# 
# After we joined weather data to the cleaned flight data, we examined the missing weather data. Wind speed, visibility, temperature and cloud mode are missing less than 2% data. We imputed these based on weather data for the same airport within a few hours. The imputation filled vast majority of the missing data. Wind angle and and atmosphere condition are missing significant amount of data, and subsequent EDA on data present (presented below) showed these two variables do not appear to have notable relationship to flight delay, therefore chose to not impute the data or include them in the models. EDA of cloud height showed possible difference between cloud height for delayed and on time flights. However, it is missing a significant amount of data, and it's also well correlated to vertical visibility and dew point temperature. Because of this, we chose to include the latter two variables as a proxy to cloud height.
# 
# We used the small dataset sampled from the full dataset including imputation to explore the relationships between weather factors 2 hours prior to a flight's scheduled departure and flight delay. 
# 
# Compare to the strong effect of previous flight, none of the weather factors have particularly strong effect on flight departure. Among the numerical variables, there are minor differences between delayed and on time lights in wind speed, vertical visibility, visibility, temperature, dew point temperature and cloud height. A correlation matrix showed that cloud height appears to be correlated to dew point temperature and visibility. For sake of minimizing the number of features to input into 
# 
# ![Figure 10](https://raw.githubusercontent.com/jying-data/imghost/master/10.png)
# 
# As for the two categorical variables, atmosphere condition is missing majority of the data even after imputing attempt, and visualizing the distribution of most common non-null values showed no obvious differences between delayed and on-time flights. Therefore, it not a good candidate for the classification model. There is a disparity between delayed and on-time flights for cloud mode 0, while the rest of the cloud modes are comparable. Mode 0 indicate clear sky, and there is an intuitive support for why this may contribute to flight timeliness. Therefore, we created a new binary variable indicating the presence or absence of clear sky.
# 
# ![Figure 11](https://raw.githubusercontent.com/jying-data/imghost/master/11.png)
# ![Figure 12](https://raw.githubusercontent.com/jying-data/imghost/master/12.png)
# 
# ## Challenges
# 
# ### Data accuracy
# 
# We identified several types of erroneous data during the EDA process. For example, there were many instances where two different flights with the same tail number having the same departure or arrival time, which is physically impossible. This resulted in duplicate entries when joining with previous flights, and was only discovered when counting distinct flight indicies and comparing against total number of rows after joining. Manual inspection of this subset of data showed that for each of these pairs flights, only departure or arrival timestamp is duplicated, never both timestamps. Therefore, we resolved this issue by ranking flight sequences based on all four flight timestamps, scheduled departure time, departure time, scheduled arrival time, arrival time. There is no way of knowing which flight among each duplicate pair is incorrect without manual inspection. This affects a thousand or so flights, which is insigificant relative to the dataset. We decided to leave these flights in the dataset with the understanding that a very small fraction of flights are known to have erroreous delay data.
# 
# Aforementioned issue with implausible values of flight duration, departure and arrival delay is another example of erroneous data. We can identify flights with implausible values without manual inspection, and therefore flights with outlier values at the very extreme ends or flights joined to such flights were omitted. This also affects an insignificant number of flights.
# 
# We note that because only date for flight departure is available, flights that arrive a day later have a mismatched date. Same issue occurs with flights that have departure or arrival delayed to past midnight. This could result in inaccurate ground preparation time and joining of previous flights. Fortunately, the number of flights affected by this is relatively minor, as there are few flights departing late at night. While scheduled flight departure and arrival times are never at midnight to avoid confusing passengers, some flights' actual departure and arrival times are recorded as 2400. The timestamp is converted to 0:00 and date is incremented by one to address this so that the resulting timestamp is compatible with standard datetime notation.
# 
# ### Scalability issue
# 
# The joining of datasets presents one of the biggest challenges to scalability. Our initial attempt to gather air traffic information was to join each flight with all other flights departing from the same airport exactly 2 to 3 hours prior and then aggregate the air traffic data based on joined flights. However, testing this approach with a small subset data at Chicago O'Hare International Airport showed the joined dataset is almost 20 times the size, because that's roughly the average hourly flights at ORD. This process would not be scalable if the dataset size was larger, as the joined data size would grow non-linearly. To circumvent this challenge, we first aggregated the hourly flight traffic information after grouping by airport and hourly block, and then joined to the main flight data based on the corresponding hourly block. This approach does not create duplicate rows from the joining and is much more efficient to run comparing to the original plan.
# 
# Joining the weather data posed the same problem, however, this process cannot be easily solved by grouping by weather station and hourly block. Since we want to join each airport with the data from up to three nearest weather stations, if we were to group and aggregate the weather data by station and hour first, join and then average the weather records for each airport, we would over-represent the stations with fewer weather readings. Additionally, weather within an hour could potentially change drastically depending on the geographical features, so an exact 2-3 hours window prior to the departure time is preferrable. Due to these reasons, we joined the unaggregated weather data to flight data. This process resulted in about 1 to 5 weather records joined to each flight record. Relative to the previous flights joining, this is less expensive. If this process has to be scaled up to dataset much larger than the ones used here, we would likely need to sacrifice some accuracy and group weather data before joining.
# 
# ### EDA Conclusion
# 
# In the EDA process, we explored the various potential features to include in our model. To improve scalability, we limit the number of relevant features to minimize the computation cost.
# 
# Additionally, we identified potential challanges to the initial data cleaning process, and deviced a solution to resolve major scalability issues.
# 
# Our data cleaning procedures were an iterative process based on discoveries made in the EDA. The final process can be summarized as the following:
# 
# 1. select important columns in flight data
# 2. join with previous flight data based on time ranking and tail number
# 3. group flight data by airport and hourly block to aggregate hourly flight traffic and delay rate at each airport
# 4. join the aggregated traffic data onto main flight table by matching the hour block 2 hours prior to scheduled departure of flight
# 5. calculate timezone and convert scheduled departure time to UTC
# 6. join a list of distinct airports to the IATA dataset for the longitude and latitude of airport
# 7. process weather data and extract relevant columns
# 8. find the nearest weather station by cross joining distinct airports with distinct weather stations and compute the distance between every possible pair
# 9. join the resulting lookup table to main flight table for nearest weather station
# 10. join weather station records by matching nearest weather station id and an exact rolling window 3 to 2 hours before flight departure
# 11. group by the distinct flight indicies and aggregate weather data by taking the median of numerical values and mode of categorical values.
# 12. impute the null values for the most important weather features
# 

# # 3. Feature Engineering
# 
# Initial feature selections were based on the EDA. We limited the features to include in the baseline model to the following:
# 
# Numerical variables:
# •	ground crew prep time
# •	hourly flight at origin airport
# •	hourly delay rate at origin airport
# •	previous departure delay
# •	previous departure delay time
# •	wind speed
# •	vertical visibility
# •	visibility
# •	temperature
# •	dew point temperature
# •	clear sky
# 
# Categorical variables:
# •	month
# •	day of week
# •	departure time block
# 
# 
# Categorical values (with more than two categories) were transformed using OneHotEncoder. There were no transformation of numerical values and binary variables because random forest can handle unscaled features. Features were vectorized as input for the models.
# 
# For feature selection/reduction, we folllowed an iterative method of running the model with full feature set, extract the Gini feature importance as an output of the Random Forest model, then reduce the feature space based on the feature importances, and finally retrain the model with the reduced feature set. The main advantage of using the Gini (impurity-based) feature ranking is that it is  built-in method in the RF model and does not add any time or additional step to the modeling pipeline. The risk with this method though, can be its tendency towards preferring features with higher cardinality. 
# 
# Based on the results of the basic random forest model, the three categorical variables were dropped from the model due to low feature importance score. Four indicator variable was created for departures in time block 0600-0659, 0700-0759, 0001-0559, and 0800-0859 since these levels of the categorical variable departure time block had a score above the cut off threshold.

# # 4. Algorithm Exploration
# 
# In addition to the baseline model (logistic regression), we also trained and evaluated ensemble random forest classifier. There were multiple reasons for this selection:
# 
# •	Robust to overfitting: given that the random forest will decide about the predicted class as the majority vote among all trees (in contrast to a single decision tree), the risk of overfitting would significantly reduce. 
# 
# •	Flexibility: since the performance of decision tree based algorithms is not dependent on the data normalization/standardization, the modeling pipeline will need less steps that ultimately helps with scalability of the algorithm. Moreover, the model will be robust to future changes and would allow to virtually include any type of data in the feature set.
# 
# •	Nonlinear classifier: When data is not linearly separable, linear classifiers give very poor results (accuracy) and non-linear classifiers like random forest (and decision trees in general) gives better results. 
# 
# •	Scalability: The implementation of ensemble learning methods including random forest and gradient boosting builds upon the original Decision Tree code, which distributes learning of single trees. Since each tree in a Random Forest is trained independently, multiple trees can be trained in parallel in addition to the parallelization for single trees. On the other hand, gradient boosting trees algorithm would train one tree at a time and thus training is only parallelized at the single tree level. Thus, RF can be significantly faster than GBT on scale. However, we should note that the larger the depth of a tree in RF (in order to improve the performance) the more time it would take to run.
# 
# •	Feature ranking: another major reason to choose random forest classifier was its inherent ability to calculate gini feature importance as part of the process, that would enormously helped with feature selection/reduction process.
# 
# **Class imbalance handling**: a major challenge with the current dataset is the severe class imbalance with only 18% of the data labeled as delayed. Several methods ahve been developed and used to handle the class imbalance among which data oversampling and/or undersampling are the most common. However, in this project we decided not to do any of these two due to the limited computational resources to perform oversampling and losing valuable information due to undersampling. Instead, we included a class weight column in the modeling process. The way this method works in random forest is that it would place a heavier penalty on misclassifying the minority class given that the classifier tends to be biased towards the majority class.
# 
# ## Helper Functions
# 

# In[4]:


from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, TimestampType
from pyspark.sql.functions import udf, explode, array, lit, col, when, isnull, isnan, count, lead
from pyspark.sql import SQLContext, Window
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import copy
import io
from io import StringIO
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator 
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler, OneHotEncoder, OneHotEncoderModel, StringIndexer, Imputer
from pyspark.ml.linalg import Vectors

from functools import reduce

from sklearn.metrics import auc, roc_curve, balanced_accuracy_score, precision_recall_curve, accuracy_score
from sklearn.metrics import precision_score, average_precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

plt.switch_backend('agg')

S3_BUCKET  = "s3://filetransfers3"
data = spark.read.option("header", "true").parquet(f"{S3_BUCKET}/cleaned/airline_weather_imputed_indicator_added")

basic_train = f"{S3_BUCKET}/cleaned/basic_train"
basic_test = f"{S3_BUCKET}/cleaned/basic_test"
selected_train = f"{S3_BUCKET}/cleaned/selected_train"
selected_test = f"{S3_BUCKET}/cleaned/selected_test"
full_data = f"{S3_BUCKET}/cleaned/full_data"

##################  HELPER FUNCTIONS ################## 

def classification_modeling(model, tr_data, ts_data):
  '''This function is used to fit the model to train data return the predictions on both train and test data'''
  # Fit the model to training data
  mdl = model.fit(tr_data)
  
  # Prediction on the training data itself
  pred_train = mdl.transform(tr_data)
  
  # Prediction on the test data
  pred_test = mdl.transform(ts_data)
  
  return pred_train, pred_test, mdl

# Zeppelin specific function for displaying matplotlib plot
def show(p):
    img = StringIO()
    p.savefig(img, format='svg')
    img.seek(0)
    print(f"%html <div style='width:600px'> {img.getvalue()} </div>")

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)
        
# area under the Precision-Recall curve
def computePrecisionRecallAUC(predictions):
    preds_proba = predictions.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))
    y_score, y_true = zip(*(preds_proba.collect()))
    pr_rc_auc = average_precision_score(y_true, y_score)
    return pr_rc_auc
    

# unified visualization of ROC & P-R in one plot
def visualizeROCPR(predictions):
    preds = predictions.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))
    y_score, y_true = zip(*(preds.collect()))
    
    fig, ax = plt.subplots(1,2, figsize = (10, 5))
    
    #visualize precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true,y_score)
    ax[0].plot(recall, precision)
    title = "2-class Precision-Recall curve"
    xlabel = "Recall"
    ylabel = "Precision"
    ax[0].set_title(title)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    
    points = CurveMetrics(preds).get_curve('roc')
    title = "ROC curve"
    xlabel = "FPR"
    ylabel = "TPR"
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]
    ax[1].plot(x_val, y_val)
    ax[1].set_title(title)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    
    plt.tight_layout()
    show(plt)

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    '''This function is used to extract the feature importances from space vector and return it as dataframe'''
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x]*100)
    return (varlist.sort_values('score', ascending = False))
    
def visualizeFeatureImportance(fi, showOnly=10):
    '''Plots the top n important features from RF model'''
    #Create a DataFrame using a Dictionary
    fi_df = fi.toPandas()
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['score'], ascending=False,inplace=True)
    if showOnly!=None:
        fi_df = fi_df[:showOnly]
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['score'], y=fi_df['name'])
    #Add chart labels
    plt.title('FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    show(plt)

# weighted metrics
def computeWeightedMetrics(predictions):
    '''The main function to calculate and print out metrics on the model prediction'''
    preds = predictions.select('DEP_DEL15','prediction').rdd.map(lambda row: (float(row['prediction']), float(row['DEP_DEL15'])))
    y_score, y_true = zip(*(preds.collect()))
    Accuracy = accuracy_score(y_true, y_score)
    Precision = precision_score(y_true, y_score, average='macro')
    Recall = recall_score(y_true, y_score, average="macro")
    F1 = f1_score(y_true, y_score, average="macro")
    ROC_AUC = roc_auc_score(y_true, y_score)

    print(f"Accuracy: {Accuracy}")
    print(f"Recall: {Recall}")
    print(f"Precision: {Precision}")
    print(f"F1 score: {F1}")
    print(f"ROC AUC: {ROC_AUC}")
    print(f"Confusion matrix:\n {confusion_matrix(y_true, y_score)}")


# ## Feature Pipeline

# In[6]:


def feature_pipeline(df, num_cols, cat_cols = None, split = [0.8, 0.2]):
    if cat_cols:
        # The index of string vlaues multiple columns
        stringIndexer = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in cat_cols]
        
        # The encode of indexed vlaues multiple columns
        encoder = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
                    outputCol="{0}_classVec".format(indexer.getOutputCol())) for indexer in stringIndexer]

        # Assemble encoded categorical and numerical variables together
        assemblerInputs = [e.getOutputCol() for e in encoder] + num_cols
        
    else:
        stringIndexer = []
        encoder = []
        assemblerInputs = num_cols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    # Vectorize all features into a single column
    cols = df.columns
    pipeline = Pipeline(stages = stringIndexer + encoder + [assembler])
    pipelineModel = pipeline.fit(df)
    df_ohe = pipelineModel.transform(df)
    selectedCols = ['features']+cols
    df_ohe = df_ohe.select(selectedCols)
    df_ohe.select('features').show(truncate = False)

    # select relevant columns for modeling: features and output
    df_ohe_ml = df_ohe.select('features','DEP_DEL15')

    # splitting the dataset into train and test
    train_final, test_final = df_ohe_ml.randomSplit(split)
    
    datasetSize  = train_final.count()
    majorityClassSize = train_final.filter('DEP_DEL15 == 0').count()
    balancingRatio = majorityClassSize / datasetSize
    train_final = train_final.withColumn("classWeights", when(train_final.DEP_DEL15 == 1, balancingRatio).otherwise(1-balancingRatio))
    print('Class balance ratio: {}'.format(balancingRatio))
    
    return df_ohe_ml, train_final, test_final


# ## Logistic Regression - Basic Model
# 

# In[8]:


# Relevant categorical and numerical variables for base model
cat_cols = ['MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK']
num_cols = ['PREV_DEP_DELAY','PREV_DEP_DEL15','PREP_TIME','HOURLY_DELAY_RATE','HOURLY_NUM_OF_FLIGHTS','PREV_CRS_ELAPSED_TIME','Temperature_imp','WindSpeed_imp','CloudMode_binary','VerticalVisibility_imp','DewPointTemp_imp']

featurized, train_final, test_final = feature_pipeline(data, num_cols, cat_cols)

featurized.write(parquet(full_data))
# train_final.write.parquet(basic_train)
# test_final.write.parquet(basic_test)


# In[9]:


train_final = spark.read.option("header", "true").parquet(basic_train)
test_final = spark.read.option("header", "true").parquet(basic_test)

# Define the model
lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", weightCol = "classWeights", maxIter = 50)

# Fit the model to train data and predict on both train and test data
predict_train_lr, predict_test_lr,  model_lr = classification_modeling(lr, train_final, test_final)


# In[10]:


# Metric assessment

print('For train set:')
computeWeightedMetrics(predict_train_lr)

print('\nFor test set:')
computeWeightedMetrics(predict_test_lr)

visualizeROCPR(predict_test_lr)


# ## Logistic Model Hyperparameter Tuning
# 
# 

# In[12]:


# Based on the results, there is high imbalanace between FN and FP predictions. To help fill this gap, we put the default threshold a bit higher that 0.5
# The only hyperparameter to tune would be the regularization parameter

lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", weightCol = "classWeights", predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', threshold=0.6)

paramGrid_lr = ParamGridBuilder()    .addGrid(lr.regParam,[0.0, 1.0])     .addGrid(lr.maxIter, [100])     .build()
    
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid_lr, evaluator= MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction"), numFolds=3)

cvModel_lr = cv.fit(train_final)

best_model_lr = cvModel_lr.bestModel

predict_train_cv_lr = cvModel_lr.transform(train_final)
predict_test_cv_lr  = cvModel_lr.transform(test_final)

# print the parameters for the best model
cvModel_lr.bestModel.extractParamMap().values()


# In[13]:


# Metric assessment
print('For train set:')
computeWeightedMetrics(predict_train_cv_lr)

print('\nFor test set:')
computeWeightedMetrics(predict_test_cv_lr)


# ## Random Forest Basic Model and Feature Importance

# In[15]:


train_final = spark.read.option("header", "true").parquet(basic_train)
test_final = spark.read.option("header", "true").parquet(basic_test)

# Define the model
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features",  predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=8, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=60, featureSubsetStrategy='auto', subsamplingRate=1.0, leafCol='', minWeightFractionPerNode=0.0, weightCol="classWeights", bootstrap=True) 

# Fit the model to train data and predict on both train and test data
predict_train_rf, predict_test_rf,  model_rf = classification_modeling(rf, train_final, test_final)


# In[16]:


# Metric assessment
print('For train set:')
computeWeightedMetrics(predict_train_rf)

print('\nFor test set:')
computeWeightedMetrics(predict_test_rf)

visualizeROCPR(predict_test_rf)


# In[17]:


fi = ExtractFeatureImp(model_rf.featureImportances, full_data, "features")
z.show(fi[['name', 'score']])

fi_ = spark.createDataFrame(fi)

# Visulaize the top 10 most important features
visualizeFeatureImportance(fi_, 10)


# # 5. Algorithm Implementation
# 
# We plan to train Random Forest models with gini impurity measure. Since a Random Forest is an ensemble of decision trees, we will explain how a hypothetical decision tree can be constructed using the Classification And Regression Tree (CART) training algorithm. CART algorithm recursively splits the training set into two subsets using a single feature _k_ and a threshold \\( t_k \\). The optimal split point can be found by minimizing the following cost function:
#  $$ J(K,t_{k}) = \frac{m_l}{m}G_l + \frac{m_r}{m}G_r   $$
# where \\( G_l \\) and \\( G_R \\) measures the impurity of the left/right subset, \\( m_l \\) and \\( m_r \\) is the number of instances in the left/right subset.
# We are planning to use gini impurity measure since it is slightly faster than other impurity measures such as entropy. The formula for calculating the gini impurity \\( G(k) \\) for a given subset k is shown below:
#  $$ G(k) = \sum_i^c P(i) * (1-P(i)) = 1 - \sum_i^c P(i)^2 $$
#  where \\( P(i) \\) is the probability of a data point being classified to class i,
# 
# Once CART algorithm successfully splits the training set in two using the optimal \\( k \\) and \\( t_k \\), it then recursively splits the subsets using the same logic until it reaches the maximum depth. It also stops splitting if other stopping criteria are met.
# 
# Let's apply CART algorithm to construct a hypothetical decision tree. Suppose we have the following toy set:  
# 
# | PreviousDelay | Temperature | ClearSky |  Delay     |
# |---------------|-------------|----------|--------------|
# | Yes           | Hot         | No      | Yes          |
# | Yes           | Cold        | Yes     | Yes          |
# | Yes           | Cold        | No      | Yes          |
# | Yes           | Cold        | Yes     | No           | 
# | No            | Cold        | No      | No          |
# 
# It contains three predictor variables and one response variable. Predictor variables are `PreviousDelay`, `Temperature` and `ClearSky`. The response variable is `Delay`. 
# 
# Let's find the optimal split point by computing gini impurity for each feature.  
# 
# - PreviousDelay
# \\( G(PreviousDelay=Yes) = 1 - (\frac{3}{4})^{2} - (\frac{1}{4})^{2} = 0.375 \\)
# \\( G(PreviousDelay=No) = 1 - (\frac{0}{1})^{2} - (\frac{1}{1})^{2} = 0 \\)
# \\( G(PreviousDelay) = \frac{4}{5}*0.375 + \frac{1}{5} * 0 = 0.3 \\)
# 
# - Temperature
# \\( G(Temperature=Hot) = 1 - (\frac{1}{1})^{2} - (\frac{0}{1})^{2} = 0 \\) 
# \\( G(Temperature=Cold) = 1 - (\frac{2}{4})^{2} - (\frac{2}{4})^{2} = 0.5 \\) 
# \\( G(Temperature) = \frac{1}{5}*0 + \frac{4}{5}*0.4 = 0.4 \\)
# 
# - ClearSky
# \\( G(ClearSky=Yes) = 1 - (\frac{1}{2})^{2} - (\frac{1}{2})^{2} = 0.5  \\) 
# \\( G(ClearSky=No) = 1 - (\frac{2}{3})^{2} - (\frac{1}{3})^{2} = 0.4444 \\) 
# \\( G(ClearSky) = \frac{2}{5}*0.5 + \frac{3}{5}*0.44444 = 0.5 \\)
# 
# `PreviousDelay` has the lowest gini impurity value so this feature will be used at the root node as shown below:
# 
# ![](https://raw.githubusercontent.com/drminix/images/master/toyexample_decisiontree_1.png)
# We can recursively apply the same technique until the stopping criteria is met. Let's find the optimal split point for the left subtree by computing gini impurity for each feature when `PreviousDelay==Yes`.
# 
# - Temperature
# \\( G(Temperature=Hot)  = 1 - (\frac{2}{2})^{2} - (\frac{0}{2})^{2} = 0 \\)
# \\( G(Temperature=Cold) = 1 - (\frac{1}{2})^{2} - (\frac{1}{2})^{2} = 0.5 \\)
# \\( G(Temperature) = \frac{2}{4}* 0 + \frac{2}{4} * 0.5 = 0.25 \\)
# 
# - ClearSky
# \\( G(ClearSky=Yes) = 1 - (\frac{1}{2})^{2} - (\frac{1}{2})^{2} = 0.5 \\)
# \\( G(ClearSky=No) = 1 - (\frac{2}{2})^{2} - (\frac{0}{2})^{2} = 0 \\)
# \\( G(ClearSky) = \frac{2}{4}*0.5 + \frac{2}{4} * 0 = 0.25 \\)
# 
# Two features have the same gini impurity value of 0.25. If we pick `Temperature`, the resulting tree looks like this:
# ![](https://raw.githubusercontent.com/drminix/images/master/toyexample_decisiontree_2.png)
# 
# 
# Since the right subtree is pure, we don't need to process this subtree. Final decision tree is shown below:
# 
# ![](https://raw.githubusercontent.com/drminix/images/master/toyexample_decisiontree_3.png)
# 
# 
# 
# 
# <br/>
# 

# ## Random Forest with Additional Feature Selection

# In[20]:


# Reduced feature set
relevant_features = ['PREV_DEP_DELAY','PREV_DEP_DEL15','PREP_TIME','HOURLY_DELAY_RATE','HOURLY_NUM_OF_FLIGHTS','PREV_CRS_ELAPSED_TIME','Temperature_imp','WindSpeed_imp','CloudMode_binary','VerticalVisibility_imp','DewPointTemp_imp', 'BLOCK6', 'BLOCK7', 'BLOCK8', 'BLOCK0']


_, train_final, test_final = feature_pipeline(data, relevant_features)

# train_final.write.parquet(selected_train)
# test_final.write.parquet(selected_train)


# In[21]:


train_final = spark.read.option("header", "true").parquet(selected_train)
test_final = spark.read.option("header", "true").parquet(selected_test)

# Define the model
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features",  predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=8, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=60, featureSubsetStrategy='auto', subsamplingRate=1.0, leafCol='', minWeightFractionPerNode=0.0, weightCol="classWeights", bootstrap=True) 

# Fit the model to train data and predict on both train and test data
predict_train_rf, predict_test_rf,  model_rf = classification_modeling(rf, train_final, test_final)


# In[22]:


print('For train set:')
computeWeightedMetrics(predict_train_rf)

print('\nFor test set:')
computeWeightedMetrics(predict_test_rf)

visualizeROCPR(predict_test_rf)


# ## Random Forest with Reduced Datasize

# In[24]:


half_data = data.sample(False, 0.5)

_, train_final, test_final = feature_pipeline(half_data, relevant_features) 


# In[25]:


# Define the model
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features",  predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=8, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=60, featureSubsetStrategy='auto', subsamplingRate=1.0, leafCol='', minWeightFractionPerNode=0.0, weightCol="classWeights", bootstrap=True) 

# Fit the model to train data and predict on both train and test data
predict_train_rf, predict_test_rf,  model_rf = classification_modeling(rf, train_final, test_final)


# In[26]:


print('For train set:')
computeWeightedMetrics(predict_train_rf)

print('\nFor test set:')
computeWeightedMetrics(predict_test_rf)

visualizeROCPR(predict_test_rf)


# In[27]:


quarter_data = data.sample(False, 0.25)

_, train_final, test_final = feature_pipeline(quarter_data, relevant_features)


# In[28]:


# Define the model
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features",  predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=8, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=60, featureSubsetStrategy='auto', subsamplingRate=1.0, leafCol='', minWeightFractionPerNode=0.0, weightCol="classWeights", bootstrap=True) 

# Fit the model to train data and predict on both train and test data
predict_train_rf, predict_test_rf,  model_rf = classification_modeling(rf, train_final, test_final)


# In[29]:


print('For train set:')
computeWeightedMetrics(predict_train_rf)

print('\nFor test set:')
computeWeightedMetrics(predict_test_rf)

visualizeROCPR(predict_test_rf)


# # 6. Conclusion
# report results and learnings for both the ML as well as the scalability.
# 
# The following is a table of model performances on the testing set. Performance on the training set are highly similar so overfitting is unlikely. Tuning the hyperparameter based on cross validation and additional feature selections based on the basic random forest model's feature importance both improved performance. For the random forest model with reduced features, reducing the dataset size down to 50% or 25% of full dataset size have negligible decrease in performance. Therefore, the effect of the predictors are fairly consistent and robust to sample sizes.
# 
# 
# | Model                     | F1-Score | ROC AUC | Recall | Precision | Accuracy |
# |---------------------------|----------|---------|--------|-----------|----------|
# | Logistic Regression       | 0.679    | 0.709   | 0.709  | 0.664     | 0.776    |
# | LR - hyperparameter tuned | 0.708    | 0.707   | 0.707  | 0.709     | 0.826    |
# | Random Forest (maxD =10 ) | 0.709    | 0.732   | 0.732  | 0.695     | 0.810    |
# | Random Forest (maxD =15 ) | 0.716    | 0.677   | 0.677  | 0.829     | 0.868    |
# | RF - features reduced     | 0.710    | 0.734   | 0.695  | 0.734     | 0.810    |
# | RF - 50% data             | 0.709    | 0.733   | 0.733  | 0.694     | 0.810    |
# | RF - 25% data             | 0.708    | 0.733   | 0.733  | 0.693     | 0.810    |
# 
# In terms of scalability, the random forest model with reduced features took 181 seconds to train on the full dataset, 119 seconds to train on 50% of the dataset, and 77 seconds to train on 25% of the dataset. The training time is roughly linear, suggesting this approach is scalable to larger datasets.
# 
# ![time](https://raw.githubusercontent.com/jying-data/imghost/master/RF%20Model%20Training%20Time%20vs.%20Size%20of%20Data.png)
# 

# # 7. Application of Course Concepts
# 
# - When preparing the input feature set for the model, the categorical features were converted to one-hot vector representation. This is necessary since machine learning algorithms assume that two nearby values are more similar than two distinct values. One hot encoding solves this problem by creating one binary attribute per category in the vector.
# - Since we are dealing with a big dataset, we tried to write the optimal code that minimizes unnecessary the shuffles and I/O operations to improve the performance. We also tried to make use of `cache()` function whenever necessary so that we do not re-evaluate the same DAG and subsequent action can reuse the cached data from the memory. 
# - Since Spark performs in-memory computation, it is an ideal candidate for iterative machine learning algorithms since the same cached data can be re-used during multiple iterations. On the other hand, Hadoop MapReduce processing engine re-reads the same data from the HDFS filesystem at every iteration so it is much slower than Spark.
# - Logistic regression model is trained using an iterative solver(gradient descent algorithm). Gradient descent algorithm will give the global minimum since the logistic regression loss function is proven to be a convex function. Using a direct solver like OLS would not be feasible since the matrix might be too big to fit on memory or not-invertible.
# - Random forest is an example of ensemble learning methods to balance bias and variance. It's robust to overfitting. The maximum depth of trees in RF model is an example of model complexity.
# 

# 
