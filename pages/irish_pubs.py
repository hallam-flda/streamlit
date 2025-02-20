import streamlit as st
import pandas as pd
import requests

st.title("Irish Pubs of Europe")

st.markdown("""
<p> When travelling Europe for the first time in 2014, 
    one of my good friends and I came across many an Irish Pub.
    Having only travelled at younger than drinking age, these were a fairly new phonomena to me.

    
            """, unsafe_allow_html=True)

st.write("""First I need to find a data set for a list of all major cities in the world. Ideally with some co-ordinates
         to calculate distances later.""")




code_block_1 = """
# Need to return a list of search IDs per lat 
# and long of all major cities in the world
# using data from https://simplemaps.com/data/world-cities.

city_data = pd.read_csv("data/data/irish_pubs/worldcities.csv")
city_data_len = f"There are {len(city_data.city)} cities in this data set"
st.write(city_data_len)
"""

# Display the code with comments
st.code(code_block_1, language="python")

# Execute the code so its output (city_data_len) also appears
exec(code_block_1)

st.markdown("""
<p> That's a lot of cities! From the sample below we can see that there are quite a few minor towns/cities that would be unlikely to have an irish pub. Furthermore there are plenty of non-european cities that will need excluding.
""", unsafe_allow_html=True)

st.dataframe(city_data.sample(10))


st.markdown("""Given we have the latitude and longitude of each city we can exclude any that fall outside of a pre-specified box for europe and any town with a population less than 50,000.</p>            
""", unsafe_allow_html=True)


code_block_2 = """
filtered_data = city_data[(35 < city_data['lat']) & (city_data['lat'] < 71) & (-25 < city_data['lng']) & (city_data['lng'] < 60) & (city_data['population'] > 10000)]
countries = filtered_data["country"].unique()

st.subheader("List of Countries in 'Europe' Map Box")
st.write(", ".join(countries))
"""

st.code(code_block_2)
exec(code_block_2)

st.subheader("Reducing Size of Dataset")

st.markdown("""
The API used to gather place information from Google Maps based on a search term is expensive and so some countries
are removed on the basis that they are either unlikely to have an Irish Pub or they aren't really part of mainland Europe.

Furthermore, United Kingdom has been removed (as it is my home country) as has the Republic of Ireland as technically every pub should
return when searching for 'Irish Pubs'.
            
Below we can see some summary statistics for the remaining countries.
            """)

code_block_3 = """
select_countries = ['Russia','Turkey','Iran','United Kingdom','Algeria',
                    'Azerbaijan','Syria','Iraq','Isle Of Man','Ireland','Tunisia','Turkmenistan',
                   'Morocco','Kazakhstan','Uzbekistan','Vatican City']

select_cities = filtered_data[~filtered_data.country.isin(select_countries)]
st.write(select_cities.groupby("country").describe())
"""

st.code(code_block_3)
exec(code_block_3)

st.markdown("""
Finally, some of these countries have a very high count of towns and cities with a population in excess of 10,000.
Therefore we can filter to only return the top 10 most populous cities of each country. There is a risk we exclude
some locations with Irish Pubs but as previously mentioned, the costly API calls prevent us from checking them all.
            """)

code_block_4 = """
city_sample = select_cities.sort_values(['country','population'], ascending=False).groupby('country').head(10)
city_sample
"""

