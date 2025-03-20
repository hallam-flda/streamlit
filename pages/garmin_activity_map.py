import streamlit as st
import streamlit.components.v1 as components


st.title("Garmin Activity Map")
st.caption("Interactive map can be found [here!](https://hallam-flda.github.io/garmin_map/)")

st.header("Introduction", divider = True)
st.write(
"""
I was gifted a Garmin Forerunner 245 for my birthday in 2021 but it wasn't until I started running more frequently in 2023 that I began to accrue a great deal of geo-location data.

Being a data analyst, naturally I wanted to see how much I could do with this data. What started as an attempt to map my most frequent running routes as a heatmap ended up turning into a detailed travel and activity log of the period 01/01/2023 - 15/02/2025.

The purpose of this page is to detail the process from start to end product. I would consider this a project still in development but being conscious that it will never be 'perfect' I will work on it as and when I get the time.
"""    
)

st.image("media/garmin_map.png")

st.header("Data Collection", divider = True)

st.write(
"""
The data I'm working with was not recorded with the intention of analysing later. Simply, I started recording my runs in order to
view my progress. I started recording other activities to post on Strava with photos as a means to share my travel photos with family members who didn't
have or want to download other forms of social media.
"""    
)

st.header("Data Ingestion", divider = True)
st.write(
"""
The first step is to login to the garmin website and [request a data export](https://www.garmin.com/en-US/account/datamanagement/exportdata).
"""
)

st.subheader("Processing in Python")
st.write(
"""
In order to get the code from Garmin download format to something that can be worked with in SQL the process is as follows

- Connect to BigQuery client
- Parse .fit files into one dataframe of data
- Upload dataframe as table in BigQuery

First the packages required for this code
"""
)

st.code(
"""
import pandas as pd
from fitparse import FitFile
import os
from google.cloud import bigquery
from google.oauth2 import service_account
"""
)

st.write("Then initialising the BigQuery client with relevant project, dataset and table IDs. Since the dataset and table ID do not already exist in my project, they will be created by the running of this script.")

st.code(
"""
# Set variables
fit_directory = "fitfiles"  # Folder containing .fit files in working directory
project_id = GOOGLE_PROJECT_ID # env variable
dataset_id = "garmin"
table_id = "activities"

# Load credentials - the JSON file has been downloaded from the Google Cloud Console and stored locally.
credentials = service_account.Credentials.from_service_account_file(f'{project_id}.json')

# Initialise BigQuery client with credentials
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
"""    
)

st.write(
"""
The folder of .fit files contains 6,742 items all of which cannot be opened in a normal text editor so they need to be converted to a format that can be uploaded into BigQuery. For
this a function called extract_fit_data() has been used that creates a JSON style output and appends it to a list for each fit file.

Since there are a lot of files and I didn't know how long it would take, I added a print statement to track progress.
""")

st.code(
"""
# Function to extract data from a FIT file
def extract_fit_data(fit_file_path):
    fitfile = FitFile(fit_file_path)
    data = []

    for record in fitfile.get_messages("record"):  # Extract GPS and sensor data
        record_data = {}
        for field in record:
            record_data[field.name] = field.value
        data.append(record_data)

    return data


all_data = []


# Loop through all FIT files and extract data
for index, file in enumerate(os.listdir(fit_directory)):
    if file.endswith(".fit"):
        file_path = os.path.join(fit_directory, file)
        all_data.extend(extract_fit_data(file_path))
        print(f'working file number {index}')

df = pd.DataFrame(all_data)
"""
)

st.write(
"""
Finally the data is uploaded into BigQuery using the variables we defined at the start of the query
""")

st.code(
"""
# Convert timestamp to bigquery datetime format
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# Upload DataFrame to BigQuery
table_ref = f'{project_id}.{dataset_id}.{table_id}'
job = client.load_table_from_dataframe(df, table_ref)

job.result()
"""    
)

st.header("Data Cleaning", divider = True)
st.write(
"""
Some of the column headers returned as 'unknown', those that I could work out I relabelled with appropriate headers and then discarded those I can't make sense of.

Since there is not a unique identifier for each activity (which I could have written in the original python query) I decided to classify any activities that take place more than 6 hours apart
from one another as a new activity.

Finally, flagging each 6 hour gap and summing across all rows serves to create an activity ID that can now be used to isolate each event.
""")

st.code(
"""
CREATE TABLE garmin.activities_clean AS

WITH
  activities AS
  -- Remove any instances where postion_lat and position_long are null because we can't do
  -- anything with this anyway
  (
  SELECT
    * EXCEPT (unknown_134,
      unknown_87,
      unknown_90,
      temperature),
    LAG(timestamp,1) OVER (ORDER BY timestamp ASC) lag_timestamp
  FROM
    garmin.activities
  WHERE
    position_lat IS NOT NULL 
    AND
    position_long IS NOT NULL
  ),

  flag_large_time_difference as (
  SELECT
    *,
    TIMESTAMP_DIFF(timestamp, lag_timestamp, HOUR) time_diff_hours,
    CASE 
      WHEN TIMESTAMP_DIFF(timestamp, lag_timestamp, HOUR) >= 6 THEN 1 
      ELSE 0 
    END AS activity_end_flag
  FROM activities
  )

  ,session_flag as 
  (
  SELECT 
  *,
  sum(activity_end_flag) over (order by timestamp asc) activity_id
  FROM flag_large_time_difference 
  ORDER BY timestamp asc 
  )
 
  SELECT
    activity_id,
    cadence,
    distance,
    enhanced_altitude,
    enhanced_speed,
    fractional_cadence,
    heart_rate,
    -- Garmin stores lat, lon combinations in semicircles rather than degrees (normal)
    position_lat * (180/2147483648) position_lat,
    position_long * (180/2147483648) position_long,
    timestamp,
    lag_timestamp,
    time_diff_hours
    activity_end_flag,
    unknown_136 as avg_heart_rate,
    unknown_135
  FROM
    session_flag

  ORDER BY
    timestamp ASC

"""    
, language = 'SQL')
