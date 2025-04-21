
import streamlit as st
from streamlit_timeline import timeline
import datetime

current_year = datetime.datetime.now()

col1, col2 = st.columns([1,1])  # Adjust ratios as needed

with col1:
    st.header("Hi, I'm Hallam 👋", divider = True)
    st.markdown(f"""
            Welcome to my personal portfolio! If you are short on time and want a quick example of my work, I recommend you start with viewing my
            [Garmin Map Project](https://hallam-flda.github.io/garmin_map/) and the [associated [write-up](https://hallamcunningham.com/garmin_activity_map).
            Otherwise, please navigate using the sidebar for other projects.

            All the code for this site and other projects can be found on my [Github](https://github.com/hallam-flda), I can be contacted either via 
            [LinkedIn](https://www.linkedin.com/in/hallam-cunningham-772a27127) or Email at hallamflda@gmail.com

            This app has been created using [Streamlit](https://docs.streamlit.io) - a Python framework used for making web applications.
            Pages marked with a ⏳ are actively being worked on. 
            """,unsafe_allow_html=True)

with col2:
    st.header(" ")
    st.image("media/headshot_circle.jpg")

skills_col, learning_col = st.columns([1,1])

with skills_col:
    st.subheader("Technical Skills")#, divider = True)
    st.markdown(
    """
    - **SQL** - Advanced - Daily use for 6+ years (BigQuery, Redshift & Impala)
    - **Python** - Intermediate - Simulations, Scripting, Data Visualisation
    - **Dashboarding** - Intermediate - Streamlit, Looker & Power BI
    - **Excel** - Advanced
    """
    )

with learning_col:
    st.subheader("Currently Learning")#, divider = True)
    st.markdown(
    """
    - **Dbt** - ETL Process - see [Using Dbt for My Acitivty Map](https://hallamcunningham.com/dbt_garmin)
    - **SQL** - Clickhouse 
    - **Streamlit** - [Advanced Streamlit dashboarding](https://hallamcunningham.com/dashboard_protoype)
    """)

st.header("Professional Summary", divider = True)

st.write("""
Senior Business Intelligence Analyst with 6 years of experience in the digital gambling industry. Over 2 years of people management duties for 3 Junior Analysts. 
Passionate about learning new skills and teaching existing skills to team members. 
         
Proven track-record of project management for the wider BI department with a focus on platform migration. Experienced in communicating with
 non-technical stakeholders and senior leadership.
             
""")



# load data
f = {
    "title": {
        "text": {
            "headline": "Employment History",
            "text": "Hallam Cunningham </p>"
        }
    },
    "events": [
            {
            "start_date": {
                "year":2016,
                "month":7,
            },
            "end_date": {
                "year":2016,
                "month":9,
            },
            "text": {
                "headline": "Econometrics Intern",
                "text": """
                        <b>MINDSHARE - LONDON</b>
                        <ul>
                        <li> Gained advanced Excel experience (Vlookups, Sumifs, Pivot Tables) to cleanse and prepare data for analysis. 
                        <li> Performed Marketing Mix Modelling in R to identify the key drivers of sales from media expenditure.
                        </ul>
                        """
            }
        },

            {
            "start_date": {
                "year":2017,
                "month":9
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
        },
            {
            "start_date": {
                "year":2017,
                "month":10
            },
            "end_date": {
                "year": 2022,
                "month": 2
                },
            "text": {
                "headline": "Co-Founder",
                "text": """
                        <b> BUDMO CLOTHING LTD </b>
                        <ul>
                        <li>Handpicked and sold 2000+ items of vintage clothing over four years, balancing evening/weekend work alongside a full-time job.</li>
                        <li>Organised multiple stock-buying trips to Ukraine and built strong local business relationships.</li>
                        <li>Acquired skills in website development, stock management, customer engagement, and accounting.</li>
                        </ul>

                        """
            }
        },
            {
            "start_date": {
                "year":2018,
                "month":12
            },
            "end_date": {
                "year": 2020,
                "month": 10
                },
            "text": {
                "headline": "Business Intelligence Analyst",
                "text": """
                        <b> SKY BETTING AND GAMING - LEEDS </b>
                        <ul>
                        <li>Compiled ad-hoc analysis to tight deadlines for the Trading Department.</li>
                        <li>Developed SQL to an advanced level with large, complex data sources, often requiring significant cleansing prior to analysis.</li>
                        <li>Used data visualization tools (R <em>(ggplot2)</em>, SAP Business Objects, Power BI) to create automated KPI reporting for hundreds of stakeholders daily, including intra-day reporting for high-profile events.</li>
                        </ul>

                        """
            },
            "group":  "Sky Betting and Gaming/Flutter Entertainment"
        },
            {
            "start_date": {
                "year":2020,
                "month":10
            },
            "end_date": {
                "year": 2022,
                "month": 12
                },
            "text": {
                "headline": "Senior Business Intelligence Analyst",
                "text": """
                        <b> SKY BETTING AND GAMING - LEEDS </b>
                        <ul>
                        <li>Completed International Leadership and Management (ILM-3) course and supervised three Junior BI Analysts.</li>
                        <li>Led a proof of concept migrating from SAP Business Objects to Looker, presenting tailored demos to leadership and cross-functional teams.</li>
                        <li>Collaborated with internal and external stakeholders to transition from Impala to Google Cloud Platform, translating existing queries to BigQuery.</li>
                        <li>Served as a trusted point of contact for the BI team, working with data engineers to resolve data-related incidents.</li>
                        </ul>


                        """
            },
            "group":  "Sky Betting and Gaming/Flutter Entertainment"
        },
            {
            "start_date": {
                "year":2022,
                "month":12
            },
            "end_date": {
                "year": 2024,
                "month": 3
                },
            "text": {
                "headline": "Career Break",
                "text": """
                        My Career Break was primarily travel-focused, however, it afforded me the chance to work on some personal development goals.
                        <ul>
                        <li>Spanish: Achieved level B1 through intensive group lessons.</li>
                        <li>Python: Completed Udemy’s 100 days of Python course, developing fundamental skills.</li>
                        <li>Python: Automated data cleansing and web-scraping for a part-time job.</li>
                        </ul>

                        """
            }
        },
            {
            "start_date": {
                "year":2024,
                "month":3
            },
            "end_date": {
                "year": 2024,
                "month": 11
                },
            "text": {
                "headline": "Senior Analyst (FTC)",
                "text": """
                        <b> FLUTTER ENTERTAINMENT - LEEDS </b>
                        <ul>
                        <li>Fixed Term Contract to support GCP-to-AWS migration, focusing on the business-critical metric of Normalised Margin.</li>
                        <li>Calculated Normalised Margin efficiency improvements exceeding £15m.</li>
                        <li>Collaborated with international teams in Romania and Ireland to explain legacy metrics and methodologies of the SkyBet heritage brand.</li>
                        <li>Introduced best practices such as documentation, change logs, and version control.</li>
                        </ul>
                        """
            },
            "group":  "Sky Betting and Gaming/Flutter Entertainment"
        }
    ]
    }


for index, event in enumerate(f["events"]):
    event["id"] = index


st.header("Employment History", divider = True)

# Render the timeline
timeline(f, height=700)


