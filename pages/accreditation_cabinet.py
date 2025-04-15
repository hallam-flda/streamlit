import streamlit as st
import streamlit.components.v1 as components


st.title("Accreditation Cabinet")
st.caption("A space to keep all the nice badges you get for completing online training")


st.subheader("Clickhouse")
st.caption("Credly doesn't allow for embedding :( link [here](https://www.credly.com/users/hallam-cunningham.9e677446) if you want to validate")
st.image("media/credly.png", width = 800)


st.subheader("Dbt")

col4, col5, col6 = st.columns([1,1,1])

with col4:
        components.iframe(
        "https://credentials.getdbt.com/embed/dff728a0-039e-44c6-845a-c3c538ac99b6",
        width=400,
        height=400,
        scrolling=True
    )