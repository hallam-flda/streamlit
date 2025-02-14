
import streamlit as st
from streamlit_timeline import timeline


st.write("# Welcome")


# Adding some text
st.write("""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut 
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut 
aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
eu fugiat nulla pariatur.
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
            "headline": "Timeline",
            "text": "Timeline </p>"
        }
    },
    "events": [
        {
            "media": {
                "url": "https://picsum.photos/200/300",
                "caption": "Photo insert"
            },
            "start_date": {
                "year":2023,
                "month":12,
                "day":27, 
                "hour": 22,
                "minute": 58,
                "second": 4,
                "microsecond":  880610
            },
            "text": {
                "headline": "photo",
                "text": " placeholder  </p>"
            }
        }
        ]}


# render timeline
timeline(f, height=800)