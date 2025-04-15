import streamlit as st
import streamlit.components.v1 as components


st.title("Accreditation Cabinet")
st.caption("A space to keep all the nice badges you get for completing online training")


st.subheader("Clickhouse")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    components.iframe(
        "https://www.credly.com/badges/55b867b1-b3a3-4a76-9563-f00c279d5f5a/public_url",
        width=200,
        height=300,
        scrolling=True
    )

with col2:
    components.iframe(
        "https://www.credly.com/badges/6c173c4f-f963-416b-be64-60bebe3f7199/public_url",
        width=200,
        height=300,
        scrolling=True
    )

with col3:
    components.iframe(
        "https://www.credly.com/badges/f2fe05a6-f2a8-4296-b225-985d6b55bfaa/public_url",
        width=200,
        height=300,
        scrolling=True
    )

    
st.subheader("Dbt")

col4, col5, col6 = st.columns([1,1,1])

with col4:
        components.iframe(
        "https://credentials.getdbt.com/embed/dff728a0-039e-44c6-845a-c3c538ac99b6",
        width=200,
        height=300,
        scrolling=True
    )