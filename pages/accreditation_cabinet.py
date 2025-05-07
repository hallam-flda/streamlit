import streamlit as st
import streamlit.components.v1 as components


st.title("Accreditation Cabinet")
st.caption("A space to keep all the nice badges you get for completing online training")


st.subheader("Clickhouse")
st.caption("Credly doesn't allow direct embedding of my profile but you can validate these [here](https://www.credly.com/users/hallam-cunningham.9e677446)")


col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.image("media/accreds/getting-started-with-clickhouse.png")
    st.image("media/accreds/insert_data.png")
    st.image("media/accreds/agg_mat_views.png")

with col2:
    st.image("media/accreds/clickhouse-architecture.png")
    st.image("media/accreds/analyse_data.png")
    st.image("media/accreds/rep_and_shard.png")
    
with col3:
    st.image("media/accreds/modeling-data-for-clickhouse.png")
    st.image("media/accreds/materialised_views.png")
    st.image("media/accreds/joining_data.png")
    
st.subheader("Dbt")

col4, col5, col6 = st.columns([1,1,1])

with col4:
        components.iframe(
        "https://credentials.getdbt.com/embed/dff728a0-039e-44c6-845a-c3c538ac99b6",
        width=400,
        height=600,
        scrolling=True
    )