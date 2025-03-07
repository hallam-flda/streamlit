
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
        st.Page("pages/travel_friends.py", title = "Travel Friends"),
        st.Page("pages/irish_pubs.py", title = "Irish Pubs of Europe")
    ],
    "Return To Player Inefficiency": [
        st.Page("pages/stochastic_processes.py", title="Stochastic Processes - Introduction"),
        st.Page("pages/stochastic_processes_ii.py", title="Stochastic Processes - Stopping Times"),
        st.Page("pages/european_roulette_sims.py", title="European Roulette Simulations", icon="⏳"),
    ],
    "Personal Development" :[
        st.Page("pages/learning_journal.py", title = "Python Learning Journal", icon="🐍"),
        st.Page("pages/habit_tracker_dashboard.py", title = "Habit Tracking Dashboard", icon="⏳")
    ]

}


pg = st.navigation(pages)
pg.run()