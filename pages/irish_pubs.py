import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
import json
from urllib.parse import urlencode
import streamlit.components.v1 as components
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import os
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from streamlit_folium import st_folium
import folium




st.title("Irish Pubs of Europe")


st.markdown("""
<i> Originally written in December 2023, it looks as if Google Maps have updated their APIs since the time of writing.           
            """, unsafe_allow_html=True)

st.header("Introduction", divider = True)

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

st.code(code_block_9)
exec(code_block_9)

st.markdown("""
Quite a lot of the results appear not to be Irish Pubs. The presence of McDonald's is particularly concerning.
What I think is happening is Google will always return results even if it doesn't have any quality matches.

In order to improve the accuracy, we can look for places that contain the words 'Irish' or 'Ireland'. While this may
not be a foolproof method, it should remove erronous results like McDonald's.

The first step to doing this is joining the two datasets. We can do this through the location information.
            """)

code_block_10 = """
df2 = pd.json_normalize(df['geometry'])
unnested_df = df.join(df2[["location.lat","location.lng"]])
unnested_df.drop(columns="geometry", inplace=True)
unnested_df
st.dataframe(unnested_df.head())
"""
st.code(code_block_10)
exec(code_block_10)

st.markdown(
    """Now from this joined dataset we want to remove any locations that do not have 'bar' listed in the 'type' field"""  
)

code_block_11 = """
# find all the entries that do not list bar as one of the types
remove_list = []

for i in range(1,len(place_ids)):
    establishment = place_ids[i]['results']
    for j in range(1,len(establishment)):
        if 'bar' not in establishment[j]['types']:
            remove_list.append(establishment[j]['place_id'])

# same dataframe as the start but remove all the places that aren't of type 'bar'
pub_df = df[~df['place_id'].isin(remove_list)]

# index stays the same so need to reset in order to make the later join work
pub_df.reset_index(inplace=True, drop = True)

# unnest the dictionary for location data
df2 = pd.json_normalize(pub_df['geometry'])
# join new df onto existing df using the index (by default)
unnested_df = pub_df.join(df2[["location.lat","location.lng"]])
# drop the old nested column
unnested_df.drop(columns="geometry", inplace=True)
# drop any duplicates (possible from two search locations being with 10km of one another)
unnested_df.drop_duplicates(inplace=True)
# reset the index
unnested_df.reset_index(inplace=True, drop=True)
# display the df
unnested_df


new_place_id_list = pub_df['place_id'].tolist()
"""


st.code(code_block_11)
exec(code_block_11)

st.markdown("""
I then passed this list into the details API, only returning place_id and reviews. These will be used to further filter.
            """)

api_details_call = """
# Get the details again but this time only for bars, because this is a subset of the original
# data we only need to return place_id and reviews and we can perform an inner join later 

places_endpoint = "https://maps.googleapis.com/maps/api/place/details/json"

results_list_filtered = []

for i in range(1,len(new_place_id_list)):

    places_params = {
        "key" : MY_API_KEY,
        "place_id" : f"{new_place_id_list[i]}",
        "fields" : "place_id,reviews"
    }
    place_params_encoded = urlencode(places_params)
    details_url = f"{places_endpoint}?{place_params_encoded}"

    r4 = requests.get(details_url)
    result = r4.json()['result']
    results_list_filtered.append(result)  # Append the 'result' to the 'results_list'

"""

st.code(api_details_call)

code_block_12 = """
with open('data/data/irish_pubs/details_data_filtered.json', 'r') as f:
    results_list_filtered = json.load(f)

# reassign
d = results_list_filtered
# stackoverflow answer - enumerate tracks the index, so check the index of the current iteration is not present
# elsewhere in the list
new_list = [i for n, i in enumerate(d) if i not in d[n + 1:]]


filtering_output = f"{len(results_list_filtered)} versus {len(new_list)}"
st.write(filtering_output)
"""

st.code(code_block_12)
exec(code_block_12)

code_block_13 = """
irish_review_list = []
no_reviews = []
# if the review contains the word ireland, add the place_id to a new list
no_review_count = 0  # Move the initialization outside the loop
for i in range(1, len(new_list)):
    pub_id = new_list[i]['place_id']
    try:
        reviews = new_list[i]['reviews']
        for j in range(len(reviews)):
            review_text = reviews[j]['text'].lower().split(" ")
            if ('ireland' in review_text) or 'irish' in review_text:
                if pub_id not in irish_review_list:
                    irish_review_list.append(pub_id)
    except:
        no_review_count += 1
        no_reviews.append(new_list[i]['place_id'])

        
len(irish_review_list)  

review_filtering = f'''
There are {len(irish_review_list)}/{len(new_list)} reviews that explicitly reference ireland/irish and
      {len(no_reviews)}/{len(new_list)} places without any reviews. Instead of filtering these out it may be better to just add this as a column to the df.
      '''

st.write(review_filtering)
      """

st.code(code_block_13)
exec(code_block_13)

code_block_14 = """
irish_review = pd.DataFrame(irish_review_list, columns = ['place_id'])#,'irish_mentioned'])
new_df = unnested_df

# similar to case statement, check if the elements of place_id are in irish review and return a Y for yes...
new_df['in_irish_review'] = np.where(new_df['place_id'].isin(irish_review['place_id']), 'Y', 'N')

# drop any rows where there are no reviews
new_df.dropna(subset=['rating'], inplace=True)

st.dataframe(new_df.sample(50))
"""

st.code(code_block_14)
exec(code_block_14)

st.header("Analysis",divider = True)

st.markdown("""
Now we have the data in a format we can work with, I wanted to ask some questions
            """)

st.subheader("Where Are the Two Closest Irish Pubs?")

code_block_15 = """
lat = np.radians(new_df['location.lat'].values)
lon = np.radians(new_df['location.lng'].values)

# Number of pubs
n = len(lat)

# Compute the pairwise differences using broadcasting
lat_diff = lat.reshape(n, 1) - lat.reshape(1, n)
lon_diff = lon.reshape(n, 1) - lon.reshape(1, n)

# Haversine formula components computed vectorized
R = 6371  # Earth's radius in kilometers
a = np.sin(lat_diff / 2)**2 + np.cos(lat.reshape(n, 1)) * np.cos(lat.reshape(1, n)) * np.sin(lon_diff / 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
dist_matrix = R * c

# Exclude self-comparisons by setting the diagonal to infinity
np.fill_diagonal(dist_matrix, np.inf)

# Find indices of the minimum distance
closest_pub_1_index, closest_pub_2_index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

# Retrieve the two closest pubs
closest_pub_1 = new_df.iloc[closest_pub_1_index]
closest_pub_2= new_df.iloc[closest_pub_2_index]
min_distance_m = dist_matrix[closest_pub_1_index, closest_pub_2_index]
"""

st.code(code_block_15)
exec(code_block_15)

col1, col2 = st.columns([1,1])  # Adjust ratios as needed

with col1:
    st.dataframe(closest_pub_1)

with col2:
    st.dataframe(closest_pub_2)

st.markdown(f"""
According to the lat, long coordinates provided by Google, these two Irish Pubs are {min_distance_m:.1f}m apart. Usually this would set the alarms ringing, however,
when dropping into the street on Google Maps, it can be seen that they are neighbours (and quite possibly owned by the same person).

Even if they do have the same decor and address, from the reviews they have different interiors and different
reviews which is good enough for me.

<b>Therefore, the two closest Irish Pubs in Europe are in Bilbao, Spain.<b>
""", unsafe_allow_html=True)




# Create an iframe with the Street View embed
iframe_html = f"""
<iframe src = "https://www.google.com/maps/embed?pb=!4v1740159719757!6m8!1m7!1szsZKxU4mHn9ajkumPfxyQQ!2m2!1d43.27109869547795!2d-2.945285583336426!3f325.4624322457213!4f-9.643372395872404!5f0.7820865974627469" width = "680" height = "450" style = "border:0;" allowfullscreen = "" loading = "lazy" referrerpolicy = "no-referrer-when-downgrade" > </iframe >

"""


# Embed the iframe in the Streamlit app
components.html(iframe_html, width=800, height=600)


st.subheader("Where is the Most Isolated Irish Pub?")

code_block_15b = """
# For each pub, find the distance to the closest other pub
closest_distances = dist_matrix.min(axis=1)

# Identify which pub has the largest 'closest distance'
furthest_pub_index = np.argmax(closest_distances)
furthest_pub = new_df.iloc[furthest_pub_index]
furthest_pub_distance_km = closest_distances[furthest_pub_index]

st.write(furthest_pub)
"""

st.code(code_block_15b)

col1a, col2a = st.columns([1,1])
iframe_html_2 = f"""
<iframe
    src="https://www.google.com/maps?q=62.009078,-6.775936&z=2&output=embed"
    width="400"
    height="300"
    style="border:0;"
    allowfullscreen=""
    loading="lazy"
    referrerpolicy="no-referrer-when-downgrade">
</iframe>
"""
with col1a:
    exec(code_block_15b)
with col2a:
    components.html(iframe_html_2, width=400, height=300)

st.markdown(f"""
            'Glitnir' in the Faroe Islands is {furthest_pub_distance_km:.2f}km away from
            it's nearest fellow Irish Pub which is located in Western Norway
            """)


st.subheader("Where Are All The Irish Pubs?")

coordinates = new_df.location.lat, new_df_location.lon

m = folium.Map(location=coordinates[0], zoom_start=5)
for lat, lon in coordinates:
    folium.Marker([lat, lon]).add_to(m)

st_folium(m, width=700, height=500)

st.subheader("What Are Common Pub Names?")

st.markdown("""
After coming across a few, we noticed that pubs tend to be named with a stereotypically Irish surname such as Murphy's or Sullivan's. One way to visualise common words and phrases
is to use a word cloud, where the size of the word is proportional to its frequency in the data.
            
Are they gimmicky? Yes. Are they used anywhere other than restaurants renovated in the early 2000s? No.
            
Regardless, I came across a rather interesting tutorial on [Medium](https://medium.com/analytics-vidhya/wordcloud-based-on-image-27afacb7cf44) that I wanted to try. First the standard out-the-box wordlcloud  from the WordCloud module:
""")

code_block_16 = """
new_df['clean_names'] = new_df['name'].str.lower().str.replace('[{}]'.format(string.punctuation), '')
all_names = ' '.join(new_df['clean_names'])
words = all_names.split()
word_counts = Counter(words)
most_common_words = word_counts.most_common()

word_count_dict = {key: value for key, value in most_common_words}
cloud = WordCloud()
cloud.generate_from_frequencies(word_count_dict)

fig = plt.figure(figsize=(8, 8))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

st.pyplot(fig)
"""

st.code(code_block_16)
exec(code_block_16)

st.markdown("""
The default plot really isn't anything to write home about and the only real inference that can be
drawn from it is commonly used words are: The Irish Pub/Bar.

While I still don't believe there is any logical reason to use a wordcloud, I still wanted to see if I 
could produce something worthy of sticking on the back of a toilet cubicle door.
""")

code_block_17 = """
guinness_colour = np.array(Image.open("data/data/irish_pubs/guin.jpg"))

# create mask  white is "masked out"
guinness_mask = guinness_colour.copy()
guinness_mask[guinness_mask.sum(axis=2) == 0] = 255

# some finesse: we enforce boundaries between colors so they get less washed out.
# For that we do some edge detection in the image
edges = np.mean([gaussian_gradient_magnitude(guinness_colour[:, :, i] / 255., 2) for i in range(3)], axis=0)
guinness_mask[edges > .3] = 255

wc = WordCloud(max_words=500, mask=guinness_mask, max_font_size=40, random_state=42, relative_scaling=0)

# generate word cloud
wc.generate_from_frequencies(word_count_dict)

# create coloring from image
image_colors = ImageColorGenerator(guinness_colour)
wc.recolor(color_func=image_colors)
guinness_fig = plt.figure(figsize=(50, 50))
plt.imshow(wc)

st.pyplot(guinness_fig)
"""

st.code(code_block_17)

st.image("data/data/irish_pubs/guinness_new.png")

