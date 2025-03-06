
import streamlit as st
from streamlit_timeline import timeline
import datetime

current_year = datetime.datetime.now()

col1, col2 = st.columns([1,1])  # Adjust ratios as needed

with col1:
    st.header("Hi, I'm Hallam 👋")
    st.markdown(f"""
             
            In {current_year.year} it no longer suffices to have a standard CV, rather than claiming I have technical skills,
            I intend to use this app as a space to demonstrate them. It also serves as a space to work on personal projects that interest me and develop
            new skills.

            This app has been created using [Streamlit](https://docs.streamlit.io) - a Python framework used for making web applications.
            Pages marked with a ⏳ are still a work in progress. I could of course hide these but I think it's important that
            I show the development process as well as just the finished project.
            """,unsafe_allow_html=True)

with col2:
    st.subheader(" ")
    st.image("media/headshot_circle.jpg")



st.header("Professional Summary")

st.write("""
Senior Business Intelligence Analyst with 6 years of experience in the digital gambling industry. Over 2 years of people management duties for 3 junior analysts. 
             Passionate about learning new skills and teaching existing skills to team members. 
             Proven track-record of project management for the wider BI department with a focus on platform migration."
             
""")



# load data
f = {
    "title": {
        "media": {
            "url": "",
            "caption": "<h1>Placeholder</h1> ",
            "credit": ""
        },
        "text": {
            "headline": "Employment History",
            "text": "Timeline </p>"
        }
    },
    "events": [
            {
            "media": {
                "url": "https://content.mindshareapps.com/media/sites/90/2022/04/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20220421153153-1024x552.png",
                "caption": "London Jul 2016 - Sep 2016"
            },
            "start_date": {
                "year":2016,
                "month":7,
            },
            "text": {
                "headline": "Econometrics Intern",
                "text": """
                        <ul>
                        <li> Gained advanced Excel experience (Vlookups, Sumifs, Pivot Tables) to cleanse and prepare data for analysis. 
                        <li> Performed multivariate regression analysis in R to identify the key drivers of sales from media expenditure.
                        </ul>
                        """
            }
        },
            {
            "media": {
                "url": "https://igamingbusiness.com/wp-content/uploads/2020/08/skybet_38.jpg",
                "caption": "Photo insert"
            },
            "start_date": {
                "year":2017,
                "month":9
            },
            "text": {
                "headline": "Operations Graduate",
                "text": """
                        <ul>
                        <li>Graduate program with rotations in Customer Experience, Safer Gambling and Trading Departments.</li>
                        <li>Gained invaluable insight across multiple business areas and making personal connections with stakeholders.</li>
                        <li>Customer facing roles helped to keep focus on improving experience for the customer in future roles.</li>
                        <li>Learnt SQL (Oracle/Impala) from scratch within role during a period of database migration requiring syntax translation from existing codes.</li>
                        </ul>

                        """
            }
        }
    ]}


# render timeline
timeline(f, height=800)