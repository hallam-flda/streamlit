import streamlit as st
import pandas as pd
import plotly.express as px

#st.set_page_config(page_title="Travel Friends", page_icon="👫")


df = pd.read_csv("data/data/people_names_redacted.csv")

st.title("Travel Friends")

st.header("Introduction")
st.write("""Between 6th January 2023 and 20th July 2023 I travelled through 4 countries in South America: Peru, Chile, Argentina and Bolivia. My original plan
was to spend a few weeks in Spanish school in Lima, Peru before venturing into Patagonia with my backpack and my tent.
There were no solid plans after that, I was just going to see what felt interesting at the time.

Early on I decided against keeping a diary, this is not something I have done before and honestly it felt like
a bit of a chore. However, I did decide to keep a log of all the people I met along the way.

It is important to note that this data is only representative of people that I approached and had a meaningful interaction with
or vice-versa. Therefore, this is not necesarily indicative of the entire travelling population. Factors such as my
rudeimentary Spanish, joining English-speaking tours and generally seeking out people of a similar demographic to myself will bias this
data heavily. It is perhaps no surprise that the most common nationality of people I met is the same as mine, British.""")
            

st.header("Data Dictionary")

data_types = df.dtypes
unique_counts = df.nunique()

# Descriptions for each column
data_desc = [
    "Numbers 1-274, originally these were names but I didn't ask permission to post this data online. For the most part, the numbers match the order in which I met the person.",
    "The Country the person is from.",
    "The Age of the person in years.",
    "The Sex of the person (M or F).",
    "Who, if anybody, the person was travelling with.",
    "The Country in which I met the person for the 1st time.",
    "How did I meet the person?",
    "How many times did I meet this person over the 6 month period (Has to be a change of location from me or the person to register again.)"
]


data_dict = pd.DataFrame({
    'Type': data_types,
    'Unique Values': unique_counts
}).reset_index()
data_dict.rename(columns={'index': 'Column'}, inplace=True)
data_dict['Description'] = data_desc

st.table(data_dict)

st.header("Geography")
st.markdown("""
As mentioned in the introduction, with the exception of the Countries I visited, 
the map consists almost entirely of countries that are well-developed and have a high proportion of English sepakers.
            
Some surprises for me were 
1) Not meeting anybody from either Asia (not including Russia and Turkey) or Africa.
2) The number of Israelis I met (9) especially considering many do not have a level of English to be confident with AND groups of Israelis often stick together in hostels. This was a common complaint of those that I befriended.
3) Not meeting any Irish for the first 5 months until I was on a tour with a group of 6 of them.
            """)


# Aggregate data to count occurrences of each country
country_counts = df['country_origin'].value_counts().reset_index()
country_counts.columns = ['Country', 'Counts']

# Assuming country_counts is already defined
fig = px.choropleth(
    country_counts, 
    locations="Country",
    locationmode='country names',
    color="Counts",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Viridis,  # A more subtle color scale
    labels={'Counts':'People Met'}
)

fig.update_geos(
    visible=False,  # Turn off the default geography visibility
    showcountries=True,  # Show country borders
    countrycolor="LightGrey"  # Set country border color to light grey
)

fig.update_layout(
    title_text='Nationality Count of Travellers I Met',
    title_x=0.4,
    geo=dict(
        lakecolor='White',  # Color of lakes
        showland=True,
        landcolor='White',
        showocean=True,
        oceancolor='LightBlue',  # A subtle ocean color
        projection_type='equirectangular'
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="RebeccaPurple"
    )
)

st.plotly_chart(fig)

st.markdown("""
Below is a plot of the number of travellers I met in each country I visited.
            """)

# Aggregate data to count occurrences of each country
country_counts_met = df['country_met'].value_counts().reset_index()
country_counts_met.columns = ['Country', 'Counts']

# Assuming country_counts is already defined
fig_2 = px.choropleth(
    country_counts_met, 
    locations="Country",
    locationmode='country names',
    color="Counts",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Viridis,  # A more subtle color scale
    labels={'Counts':'People Met'}
)

fig_2.update_geos(
    visible=False,  # Turn off the default geography visibility
    showcountries=True,  # Show country borders
    countrycolor="LightGrey",
    scope='south america'  
)

fig_2.update_layout(
    title_text='Location of Meeting Place',
    title_x=0.4,
    geo=dict(
        lakecolor='White',  # Color of lakes
        showland=True,
        landcolor='White',
        showocean=True,
        oceancolor='LightBlue',  # A subtle ocean color
        projection_type='equirectangular'
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="RebeccaPurple"
    )
)

st.plotly_chart(fig_2)

st.markdown("""
Each country looks roughly proportional to the amount of time I spent in each, however, my experiences were rather different from Country to Country:
            
<ul>
            <li> In <b>Peru</b> my main activities were Spanish School and the Salkantay trek, both of which meant spending considerable time with the same people.
            <li> In <b>Chile</b> I was primarily hiking and camping solo, however, I did depend on hitchhiking a lot for transportation.
            <li> <b>Argentina</b> was the country I spent by far the most time in and through my girlfriend met many locals.
            <li> I spent the least time in <b>Bolivia</b> but did take another two weeks of Spanish Lessons which invariably leads to making connections.
                <ul>
                    <li> Bolivia was also the last country I visited and the country where I met a lot of the same people I had earlier in the trip all of whom would be attributed to another country.
                </ul>
</ul>
            """, unsafe_allow_html=True)

st.subheader("Appendix - Full Dataset")


#df