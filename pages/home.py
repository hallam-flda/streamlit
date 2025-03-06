
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
            "start_date": {
                "year":2016,
                "month":7,
            },
            "start_date": {
                "year":2016,
                "month":9,
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
                        My Career Break was primarily travel-focussed, however, it afforded me the chance to work on some personal development goals.
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


# render timeline
timeline(f, height=700)