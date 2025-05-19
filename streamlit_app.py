
import streamlit as st

st.set_page_config(
    page_title="Hallam Cunningham",
    page_icon="📈",
    layout = 'centered'
)

pages = {
    "Home": [
        st.Page("pages/home.py", title = "Introduction")
    ],

    "Ad-Hoc Analysis":[
        st.Page("pages/garmin_activity_map.py", title = "Garmin Activity Map"),
        st.Page("pages/travel_friends.py", title = "Travel Friends"),  
        st.Page("pages/argentina_home_advantage.py", title = "Argentinian League Home Advantage")
        #st.Page(
            # "pages/irish_pubs.py",
            # title="Irish Pubs of Europe",
            # url_path="irish_pubs"  
            # )
    ],
    
    "Personal Development" :[
        st.Page("pages/dbt_garmin.py", title = "Using Dbt for My Activity Map"),
        st.Page("pages/deploying_to_google_cloud.py", title = "Hosting This Site on Google"),
        st.Page("pages/accreditation_cabinet.py", title = "Accreditation Cabinet"),
        st.Page("pages/learning_journal.py", title = "Python Learning Journal", icon="🐍")
       # st.Page("pages/habit_tracker_dashboard.py", title = "Habit Tracking Dashboard", icon="⏳")
    ],

    "Return To Player Inefficiency": [
        st.Page("pages/rtp_intro.py", title="Introduction"),
        st.Page("pages/stochastic_processes.py", title="Stochastic Processes - Introduction"),
        st.Page("pages/stochastic_processes_ii.py", title="Stochastic Processes - Stopping Times"),
        st.Page("pages/european_roulette_sims.py", title="European Roulette Simulations I")#,
        #st.Page("pages/stopping_time_plot.py", title="Stopping Time Plot")
    ],
    "Football Dashboard": [
    st.Page("pages/dashboard_prototype.py", title="Dashboard Prototype", icon = "⏳")
    ]
}


pg = st.navigation(pages)
pg.run()