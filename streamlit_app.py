
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


pages = {
    "Home": [
        st.Page("pages/home.py", title = "Introduction")
    ],
    "Maths Theory": [
        st.Page("pages/stochastic_processes.py", title="Stochastic Processes")
    ],
    "Projects" :[
    ],
    "Personal":[
        st.Page("pages/travel_friends.py", title = "Travel Friends")
    ]

}


pg = st.navigation(pages)
pg.run()