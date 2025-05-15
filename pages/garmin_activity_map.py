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

st.image("media/garmin/garmin_map.png")

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
this a function called `extract_fit_data()` has been used that creates a JSON style output and appends it to a list for each fit file.

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
The first step is to take a look at the data uploaded from the Python import...
"""
)

st.image("media/garmin/select_star_garmin.png")

st.write(
"""
Some of the column headers returned as 'unknown', those that I could work out I relabelled with appropriate headers and then discarded those I couldn't make sense of. The schema description
tab in BigQuery provides a handy space to add notes for each column.
"""
)

st.image("media/garmin/activities_schema.png")

st.write("""
Any rows that have a null lat or long field are removed as they won't be useful for this exercise.
         
Temperature is always null as my model of Garmin does not have that capability.
         
Since there is not a reliable unique sequential identifier for each activity I decided to classify any activities that take place more than 6 hours apart
from one another as a new activity.

Finally the lat, lon columns have to be transformed from semi-circle units (as used in fit files) to regular degree based lat, lon pairs.
""")

st.code(
"""
CREATE TABLE garmin.activities_clean AS

WITH
  activities AS
  -- Remove any instances where position_lat and position_long are null because we can't do
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

st.header("Derived Columns", divider = True)

st.write("""
Now that the data contains only relevant fields and non-null rows, some new columns are required for use with the [kepler.gl](https://kepler.gl/) framework.
""")

st.code(
"""

create table garmin.activities_routed as

WITH
  base_table_with_lags AS (
  SELECT
    *,
    MAX(timestamp) OVER (PARTITION BY activity_id) end_time,
    MIN(timestamp) OVER (PARTITION BY activity_id) start_time,
    LEAD(timestamp) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_timestamp,
    LEAD(position_lat) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_lat,
    LEAD(position_long) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_long,
    MAX(distance) OVER (PARTITION BY activity_id) end_distance,
    MIN(distance) OVER (PARTITION BY activity_id) start_distance
  FROM
    `tonal-run-447413-c0.garmin.activities_clean`
     )

, duration_metrics as 
(  
SELECT
  activity_id,
  avg_heart_rate,
  cadence,
  enhanced_altitude,
  distance distance_meters,
  start_time,
  end_time,
  position_lat,
  position_long,
  lead_lat,
  lead_long,
  end_distance,
  timestamp,
  TIMESTAMP_DIFF(lead_timestamp, timestamp, SECOND) time_increment,
  TIMESTAMP_DIFF(end_time, start_time, SECOND) adjusted_activity_duration_seconds,
  TIMESTAMP_DIFF(end_time, start_time, MINUTE) adjusted_activity_duration_minutes,
  end_distance/TIMESTAMP_DIFF(end_time, start_time, SECOND) adjusted_speed_ms
FROM
  base_table_with_lags
)

select
*,
sum(time_increment) over (partition by activity_id) activity_duration_seconds,
(sum(time_increment) over (partition by activity_id))/60 activity_duration_minutes,
end_distance/sum(time_increment) over (partition by activity_id) avg_speed_ms,
60 / ((end_distance/sum(time_increment) over (partition by activity_id))*3.6) pace_mins_per_km

from
duration_metrics

order by activity_id asc, timestamp asc

""" , language = "SQL"   
)

st.write(
"""
First, time and location columns are 'maxed and minned' to help calculate total distance and time of each activity. The lead of these columns is also taken to create incremental windows.
This will be particularly important for plotting these exercises as the 'line' type in Kepler.

Then the `TIMESTAMP_DIFF()` function is used on the columns created in the previous select to give some important duration metrics.

Finally an activity duration is re-calculated using the sum between each timestamp. The purpose of this is to differentiate between chronological time and time spent actually
on activity in the instance of paused events. In writing this up I realise that this will be the same as the total timestamp difference so to fix this I will need to exclude any abnormally
long gaps in entries (more than 20-30s). Something else to add to the issues log!
"""    
)

st.header("Plotting Garmin Activities", divider = True)

st.write(
"""
Plotting the data in Kepler is remarkably simple. Once uploading the data from `garmin.activities_routed` as a CSV exported from BigQuery, all that is required is to choose the plotting
method and map the correct columns to the required fields in the Kepler UI. 

Below is a screenshot of my regular run routes, most of which start in the Kirkstall area where I used to live.
"""    
)

st.image("media/garmin/local_routes.png")

st.write(
"""
This is all I was really hoping for when I started this project, a way to visualise all of my common routes along with some interesting stats about pace, heart rate etc.

However, when zooming out to view on a global scale, it isn't immediately obvious to the viewer where to zoom into.
"""
)

st.image("media/garmin/world_view_empty.png")

st.write(
"""
During my career break in 2023, I travelled the lower half of South America pretty comprehensively, making a conscious effort to [keep a physical record of the people I met.](hallam-flda.streamlit.app/travel_friends)
I also recorded most of my physical activities, however, as it is a truly enormous continent, some of these plots are quite small and distant from one another unless you zoom in.

This gave me an idea: Can I populate the gaps with my travel in-between locations?
"""
)

st.header("Plotting Transit", divider = True)

st.write(
"""
Returning to the `activities_clean` dataset in BigQuery, I wanted to create a travel log that would flag any significant change in distance between two recorded garmin activities and treat this as
a change in location.
"""    
)

st.code(
"""
WITH
  start_end_times AS (
  SELECT
    *,
    MIN(timestamp) OVER (PARTITION BY activity_id) min_act_time,
    MAX(timestamp) OVER (PARTITION BY activity_id) max_act_time
  FROM
    `garmin.activities_clean` ),
    
  lagging_values AS (
  SELECT
    *,
    LAG(activity_id) OVER (ORDER BY timestamp) lag_activity_id,
    LAG(position_lat) OVER (ORDER BY timestamp) lag_lat,
    LAG(position_long) OVER (ORDER BY timestamp) lag_long
  FROM
    start_end_times
  WHERE
    min_act_time = timestamp
    OR 
    max_act_time = timestamp ),

  distances AS (
  SELECT
    *,
    -- built in BigQuery function for calculating distance between lat,long pairs
    ST_DISTANCE( ST_GEOGPOINT(position_long, position_lat), ST_GEOGPOINT(lag_long, lag_lat) ) / 1000 AS distance_km
  FROM
    lagging_values
  WHERE
    -- there is a change in the activity_id
    activity_id <> lag_activity_id 
    )

SELECT
  *
FROM
  distances
WHERE
  distance_km > 30 -- want to indicate when I've moved a decent distance
ORDER BY
  timestamp ASC
"""  , language = "SQL" 
)

st.write(
"""
This query returns a dataset with 74 rows and lat, lon pairs between changes of more than 30kms. The combination of BigQuery's `ST_DISTANCE()` and `ST_GEOPOINT()` functions made
this far easier than I was anticipating.

Returning to plot this on kepler, the arc plotting method provides a good way to show distant start and end points.
"""    
)

st.image("media/garmin/first_travel_log.png")

st.write(
"""
However, when zooming into some of my more travelled destinations such as Leeds, where I lived and worked and Sussex, where my family live, the plot becomes messy and looks almost as if
I had been flying between the two locations.
"""    
)

st.image("media/garmin/messy_arcs.png")

st.write(
"""
To get around this I decided to trawl through the log manually and add a travel method to each row. While I trust my memory for the most part, adding this data is prone to human error
(either with data entry or misremembering.) This is not something I would add if accuracy was critical.
"""   
)

st.image("media/garmin/colour_coded_arcs.png")

st.write(
"""
This is much better, however, I still don't like the fact that every change in location looks like a flight.

So I had another idea: How can I plot my land route, even though I have not recorded this data?
"""   
)

st.header("Google Routes API", divider = True)

st.write(
"""
Since I have a log of lat, lon pairs for departure destination and arrival destination, I have all of the information required to plot a route in Google Maps. While there is no
guarantee the route I took is the same as Google's choice of route, I think it still gives a better idea of any travel that took place across land.

In order to use the [Google Routes API](https://developers.google.com/maps/documentation/routes), some more set-up is required to get the data in a format that can be posted to the API endpoint.

Rather than reimport all the code here, I have appended the notebook I used to download from BigQuery, pass it to the Google API and save the response back in a BigQuery table. It took me a few
attempts to get the dataframe in the correct format but working with a notebook directly in the BigQuery console makes the whole process a lot quicker than switching from IDE to console.
"""    
)

with open("static/Non_flight_transit_routing.html", "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=800, scrolling = True)

st.code(
"""
create table garmin.ranked_journeys as 

SELECT
  *,
  lead(longitude) over (partition by activity_id order by step asc) lead_long,
  lead(latitude) over (partition by activity_id order by step asc) lead_lat
FROM
  `tonal-run-447413-c0.garmin.non_flight_journeys`

  order by activity_id asc, step asc
""", language = 'SQL'
)

st.write(
"""
Once uploaded to BigQuery after leading the lat and lon positions the data can be exported as a CSV ready to use in the kepler UI.

The output of this API response is around 190,000 rows of routing data for my non-flight transit. This is a lot of data and bloats the final HTML file to a size of 91MB. This means
that I am not able to host directly within streamlit and it can fail to load on mobile devices. Another improvement would be to remove some intermediate steps especially in the longer journeys.
"""    
)

st.header("Improved Transit Plotting", divider = True)

st.write(
"""
Now with the complete set of journey co-ordinates, the entire route can be plotted along with some extra information on the tooltip such as journey duration and distance.

For the same region as the previous example, a much clearer picture of the actual route I took is now displayed.
"""    
)

st.image("media/garmin/improved_land_transit.png")

st.header("Potential Improvements", divider = True)

st.write(
"""
Overall I think the map currently hosted on Github passes the mark for a minimum viable product, however, there are some changes that I think would improve this map.
"""    
)

st.subheader("File Size - Complete ✅")

st.write(
"""
Currently the file size is 91MB which is massive for an HTML file. This can be greatly reduced by removing some of the more detailed land journeys or simply deleting every other row.
The segment level detail is a bit overkill especially considering no actual data (such as heart rate or cadence) is recorded as with my recorded garmin data.
"""
)
st.caption("Update - 21/03/2025")

st.write("""
I have now deleted any obsolete columns and reformatted the number formats in excel. This has reduced file size down to 43.5MB. This is still large and the idea of deleting a portion of the rows
makes sense but it is a very good low-effort start.
""")
st.caption("Update - 25/03/2025")
st.write("""
I have kept every 4th line for physical activities and 5th line for land transit. This was a bit annoying because it meant I had to re-lead the coordinates in SQL but the file size is now
sitting at 9.1MB. I also removed the first and last 5% of my activities as to disguise start and end points.
"""    
)

st.subheader("Travel Accuracy")

st.write(
"""
Transit is plotted between two consecutive activities in my Garmin history. For this to be a realistic representation of my travel I would have to record activities immediately prior to
leaving one location and immediately on arrival at the next. This clearly isn't always going to be the case. For the most part this doesn't really matter, it is only an indication of
where I have travelled in this time period.

One potential 'quick win' would be to find the nearest airport for each of the start and end points of my flights. At the moment I returned from Buenos Aires to Southend which is not a real
flight path sadly.

I have also used roads to plot train journeys. Again, this could be fixed with some agonising through the Google API docs but accuracy was never the top priority with this project.
"""    
)

st.subheader("Auto Updating - In Progress ⏳")

st.write(
"""
The map was constructed entirely from one export of my garmin files up to 15th Feb 2025. There is a point to be made that constantly posting your exact whereabouts on the internet might not
be the smartest move so any auto-updating should probably exclude the month previous. That being said it would be a nice improvement to have the map update itself with frequent pulls of
garmin data using the garminconnect API. I may revisit this in the future but for the time being I would repeat the process every 3-6 months or so. 
"""    
)

st.caption("Update - 01/04/2025")

st.write(
"""
Having recently taken the [Dbt Fundamentals training](https://learn.getdbt.com/learn/course/dbt-fundamentals) which I would strongly recommend for anyone trying to automate a data flow, I have now set up a workflow so I only need
to append new data to the original `garmin.activities` dataset. Then I run the job in Dbt and it will output the end files I need to update my map; not quite auto-updating but removes
a fair amount of the manual lifting.
"""
)

