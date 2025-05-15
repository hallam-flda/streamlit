import streamlit as st
import inspect

st.title("Python Learning Journal")

st.header("Introduction")

st.write(
    """
    A space to add any useful tricks that I would otherwise forget.
    """
)

st.header("Utilising Google Cloud Platform for SQL & Python")

st.subheader("Intro")
st.write(
"""
For most of the code on this site I've used a combination of self-written and AI generated python output.
Of course AI will always attempt to generate code in the format you ask for but at times I've been using the wrong
tool for the job. In particular, when calculating rolling margin for the European Roulette Sims, I spent an entire
afternoon trying to find the silver-bullet prompt when I should really have just used my existing SQL knowledge.

Part of my reluctance to use SQL is the extra step of maintaining and hosting a database that can be easily accessed.
Given I have experience using BigQuery I have started using it to host personal-project databases. Using cloud services
has the benefit of keeping my data accessible in case of laptop theft (currently in Brazil!)
"""    
)

st.subheader("Garmin Use Case")
st.write(
"""
I have always enjoyed clever visualisations that utilise maps. Geospatial data comes with a whole host of compatibility challenges
with regards to plotting in Python and often requires some data-cleaning prior to pairing with mapping packages. Instead
of defaulting to Pandas, I decided to host my garmin activities data on GCP.
"""
)

st.subheader("Obtaining The Data")
st.write(
"""
For my daily steps counter used in the habits dashboard, I was able to use an unofficial garminconnect api to retrieve daily information.
Unfortunately this approach does not work for activities, so the first step is to login to the garmin website and [request a data export](https://www.garmin.com/en-US/account/datamanagement/exportdata).

Since the data is returned as a series of .fit files, they need to be converted to a format that I can upload into BigQuery. For simplicity's sake, I used a
package called [fitparse](https://github.com/dtcooper/python-fitparse) to parse these files into CSV format which I would then upload to GCP.
""")

st.subheader("Data Cleaning")
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
    # Garmin stores lat, lon combinations in semicircles rather than degrees (normal)
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
)

st.subheader("Integrating BigQuery and Python")
st.write(
"""
For ideation, I tend to use colab notebooks. I was forced into this by something going awry with my jupyter setup but it's probably for the best
as you can create colab notebooks directly in the GCP interface.
""")

st.image("media/bq_screen.png")

st.write(
"""
From here I can query my BigQuery databases directly into a notebook format to play around with the data.
"""
)

st.code(
"""
%%bigquery results
SELECT * FROM `garmin.activities_clean`
"""
)

st.write(
"""
 I have since found another way to plot my garmin data which simply requires uploading a CSV (already generated prior to uploading to BigQuery), however, this will be useful to remember for any
future queries that require a database that is being frequently updated.

Also in my return-to-player simulation work, I will eventually want to build a large dataset that would be impractical to host and upload locally.
"""    
)

st.header("Rendering Code In Streamlit")

def add_two(n):
    return n+2

x = add_two(n=5)

with st.expander("Example Function"):
    st.code("""
            def add_two(n):
                return n+2

            x = add_two(n=5)
            """)



st.markdown(
"""
So far I have found 3 ways to render and execute code in streamlit:
"""
)

st.subheader("Option 1")
st.markdown("""
Write code blocks as text and then add the text to an expander with an execution command calling the code if it needs to be executed
"""
)

st.code("""
code_block_1 = '''
def add_two(n):
    return n+2

x = add_two(n=5)
st.write(x)        
'''
        
st.code(code_block_1)

exec(code_block_1)                
""")

st.markdown(
"""
This is probably the worst way of doing it as VS Code does not recognise the variables being defined within the text variable leading to lots of error messaging in the IDE.
It also means coming up with names for every section of code which is not always easy. Finally it also prints the streamlit commands as well if we need to display the output.
"""
)

st.subheader("Example")
code_block_1 = '''
def add_two(n):
    return n+2

x = add_two(n=5)
st.write(x)        
'''

st.code(code_block_1)
exec(code_block_1) 

st.subheader("Option 2")
st.markdown("""
Option 2 is using a combination of st.code() and st.echo()
"""
)

st.code("""
with st.echo():
    def add_two(n):
        return n+2

    x = add_two(n=5)

    st.write(x)
""")

st.subheader("Example")
with st.echo():
    def add_two(n):
        return n+2

    x = add_two(n=5)
    
    st.write(x)

st.markdown("""
This is a good option but it depends on explicitly defining the code within the echo statement every time, reducing readability. We also still see the st.write() command
"""
)

st.subheader("Option 3")
st.markdown("""
Finally my preferred option is to define functions outside the scope of any with statement directly into the .py file and use inspect to get the source of the function
""")

st.code("""
import inspect

def add_two(n):
    return n+2
              
x = add_two(5)
        
st.code(inspect.getsource(add_two)+ "\\nx = add_two(5)")
st.write(x)

""")

st.subheader("Example")

st.code(inspect.getsource(add_two)+ "\nx = add_two(5)")
st.write(x)

st.markdown("""
This approach allows for more customisability with how much of the code to output, including the st.write() or st.pyplot() functions.
            """)



