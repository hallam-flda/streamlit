import streamlit as st
import pandas as pd
import requests
import json
from urllib.parse import urlencode

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
st.dataframe(city_sample)
"""
st.code(code_block_4)
exec(code_block_4)

st.header("Querying Google Maps APIs")

st.markdown("""
The Nearby Search API used only returns a maximum of 20 results per search. With the exception of London (already excluded)
I think we can assume it is unlikely any of these cities will have in excess of 20 Irish Pubs.

Some extra data prep is required to make it easier to query the API
           """)

code_block_5 = """
data_list = city_sample.apply(lambda row: {"city": row["city"], "location": (row["lat"], row["lng"])}, axis=1).tolist()
st.write(f'{data_list[0]["location"]}' + ' - These tuples of lat lng combinations can now be passed to the API')
"""

st.code(code_block_5)
exec(code_block_5)

code_block_6 = """
place_ids = []

# This is expensive to run - do not run again 

for i in range(1, len(data_list)):
   search_endpoint_trim = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
   params = {
       "key" : MY_API_KEY,
       "location" : f"{data_list[i]['location'][0]},{data_list[i]['location'][1]}",
       "radius" : 10000,
       "keyword" : "irish pub"
   }
   params_encoded = urlencode(params)
   places_url = f"{search_endpoint_trim}?{params_encoded}"

   r2 = requests.get(places_url)
   result = r2.json()
   place_ids.append(result)  # Append the 'result' to the 'results_list'   
     
"""

st.code(code_block_6)

st.markdown("""
Results from the original run of this analysis have been saved into a JSON file to avoid re-running the query against the API.
These have been loaded below and a list of place_ids to pass to a Place Details API.
            """)

code_block_7 = """
with open('data/data/irish_pubs/data.json', 'r') as f:
    place_ids = json.load(f)

id_list = []

for j in range(len(place_ids)):
    for i in range(1, len(place_ids[j]['results'])):
        #new_id = place_ids[j]['results'][i]['place_id']
        id_list.append(place_ids[j]['results'][i]['place_id'])    

st.write(id_list[:5])
"""

st.code(code_block_7)
exec(code_block_7)

code_block_8 = """
places_endpoint = "https://maps.googleapis.com/maps/api/place/details/json"

results_list = []

for i in range(1,len(id_list)):

    places_params = {
        "key" : YOUR_API_KEY,
        "place_id" : f"{id_list[i]}",
        "fields" : "name,rating,geometry,place_id,rating,url,website,reviews"
    }
    place_params_encoded = urlencode(places_params)
    details_url = f"{places_endpoint}?{place_params_encoded}"

    r3 = requests.get(details_url)
    result = r3.json()['result']
    results_list.append(result)  # Append the 'result' to the 'results_list'

with open('details_data.json', 'w') as f:
    json.dump(results_list, f)

"""

st.code(code_block_8)

code_block_9 = """
with open('data/data/irish_pubs/details_data.json', 'r') as f:
    results_list = json.load(f)

df = pd.DataFrame(results_list)

st.dataframe(df)
"""

st.code(code_block_8)
exec(code_block_9)

st.markdown("""
Quite a lot of the results appear not to be Irish Pubs. The presence of McDonald's is particularly concerning.
What I think is happening is Google will always return results even if it doesn't have any quality matches.

In order to improve the accuracy, we can look for places that contian the words 'Irish' or 'Ireland'. While this may
not be a foolproof method, it should remove erronous results like McDonald's.

The first step to doing this is joining the two datasets. We can do this through the location information.
            """)

code_block_10 = """
df2 = pd.json_normalize(df['geometry'])
unnested_df = df.join(df2[["location.lat","location.lng"]])
unnested_df.drop(columns="geometry", inplace=Trua
unnested_df
"""

