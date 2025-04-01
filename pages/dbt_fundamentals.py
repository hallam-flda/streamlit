import streamlit as st

st.title("Dbt Fundamentals")

st.subheader("Introduction", divider = True)

st.write(
"""
Increasingly it is becoming a requirement of Data Analysts to have an understanding of the end-to-end data process. In large organisations, analysts are usually served a final production quality
data table from which they can serve business insights to stakeholders. I have found bridging the gap between engineers and stakeholders can be frustrating at times, especially when, as an analyst, you don't
fully understand the ETL/ELT process yourself.

For this reason, I have followed Dbt's extremely useful fundamentals course to create a job based off the [garmin map](https://hallam-flda.streamlit.app/garmin_activity_map) workflow.
This taught me a lot on how data should be structured and the difference between development/testing/production environments, as well as the need for staging tables etc. I would recommend any 
analyst who does not have a good understanding of data engineering to take the course, even if they never need to use Dbt.
"""    
)

st.subheader("The Use Case - Garmin Data", divider = True)

st.write(
"""
The course provides a good set of example data, however, I find that the knowledge sticks better when I'm working on something I care about. When working on my garmin map visualisation, I did a lot of 
data cleaning, most of which is documented in the write up, however, it certainly was not best practise, nor did I organise the files efficiently enough to be able to reproduce easily with new data. What I needed was
a fully production-ready data workflow to take raw data and output data in the same format as kepler takes in the kepler UI.
"""    
)