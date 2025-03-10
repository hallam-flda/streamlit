
import streamlit as st
from streamlit_timeline import timeline
import datetime

current_year = datetime.datetime.now()

col1, col2 = st.columns([1,1])  # Adjust ratios as needed

with col1:
    st.header("Hi, I'm Hallam 👋")
    st.markdown(f"""
            Welcome to my personal website! The intention behind this site is a place to share some examples of my analysis as well as a space to work on personal projects that interest me and develop
            new skills.

            This app has been created using [Streamlit](https://docs.streamlit.io) - a Python framework used for making web applications.
            Pages marked with a ⏳ are still a work in progress. I could of course hide these but I think it's important that
            I show the development process as well as just the finished project.
            """,unsafe_allow_html=True)

with col2:
    st.subheader(" ")
    st.image("media/headshot_circle.jpg")



st.header("Professional Summary", divider = True)

st.write("""
Senior Business Intelligence Analyst with 6 years of experience in the digital gambling industry. Over 2 years of people management duties for 3 Junior Analysts. 
Passionate about learning new skills and teaching existing skills to team members. 
Proven track-record of project management for the wider BI department with a focus on platform migration.
             
""")



# load data
f = {
    "title": {
        "text": {
            "headline": "Employment History",
            "text": "Hallam Cunningham"
        }
    },
    "events": [
        {
            "start_date": {
                "year": 2016,
                "month": 7
            },
            "end_date": {
                "year": 2016,
                "month": 9
            },
            "text": {
                "headline": "Econometrics Intern",
                "text": """
                    <b>MINDSHARE - LONDON</b>
                    <ul>
                    <li> Gained advanced Excel experience (Vlookups, Sumifs, Pivot Tables) to cleanse and prepare data for analysis.</li>
                    <li> Performed Marketing Mix Modelling in R to identify the key drivers of sales from media expenditure.</li>
                    </ul>
                """
            }
        },
        {
            "start_date": {
                "year": 2017,
                "month": 9
            },
            "end_date": {
                "year": 2018,
                "month": 12
            },
            "text": {
                "headline": "Operations Graduate",
                "text": """
                    <b> SKY BETTING AND GAMING - LEEDS </b>
                    <ul>
                    <li>Graduate program with rotations in Customer Experience, Safer Gambling and Trading Departments.</li>
                    <li>Gained invaluable insight across multiple business areas and making personal connections with stakeholders.</li>
                    <li>Customer facing roles helped to keep focus on improving experience for the customer in future roles.</li>
                    </ul>
                """
            },
            "group": "Sky Betting and Gaming/Flutter Entertainment"
        }
    ]
}


for index, event in enumerate(f["events"]):
    event["id"] = index


st.header("Employment History", divider = True)

# Render the timeline
timeline(f, height=700)


