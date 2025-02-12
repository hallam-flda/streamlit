import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np

#st.set_page_config(page_title="Travel Friends", page_icon="👫")


df = pd.read_csv("data/data/people_names_redacted.csv")

st.title("Travel Friends")

st.header("Introduction")
st.write("""Between 6th January 2023 and 20th July 2023 I travelled through 4 countries in South America: Peru, Chile, Argentina and Bolivia. My original plan
was to spend a few weeks in Spanish school in Lima, Peru before venturing into Patagonia with my backpack and my tent.
There were no solid plans after that, I was just going to see what felt interesting at the time.

Early on I decided against keeping a diary, this is not something I have done before and honestly it felt like
a bit of a chore. However, I did decide to keep a log of all the people I met along the way.
""")

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

st.subheader("Data Collection Notes")

st.markdown("""

It is important to note that this data is only representative of people that I approached and had a meaningful interaction with
or vice-versa. Therefore, this is not necessarily indicative of the entire travelling population. Factors such as my
rudimentary Spanish, joining English-speaking tours and generally seeking out people of a similar demographic to myself will bias this
data heavily. 

What constitutes as a meaningful interaction? Simply whether I remembered the person when it came to updating the log, usually 2-4 weeks from the date of us meeting.
           """)

st.header("Who Did I Meet on Average?")

st.markdown("""
Taking a form of average for each column, the _average_ person I met was:         
            """)

st.write(f"""
Most Common Country = {df.country_origin.mode()[0]} \n
Median Age = {df.age.median():.2f} \n
Most Common Sex = {df.sex.mode()[0]} \n
Travelling As = {df.travelling_as.mode()[0]} \n
Most Likely to Meet in/on = {df.location_met.mode()[0]} \n
How Often Did We Meet = {df.times_met.mode()[0]}
""")

st.markdown("""
While taking various forms of averages across different categories isn't always likely to yield a realistic answer, I did in fact meet 2 people who match this description.         
            """)

st.title("Geography")
st.markdown("""
When plotting the nationalities of fellow travellers on a Cloropleth Map, it can be seen that people I met tend to be from countries
that are well-developed economically and culturally similar to the UK.
            
Some surprises for me were 
1) Not meeting anybody from either Africa or Asia (not including Russia and Turkey).
2) The number of Israelis I met (9) especially considering many do not have a level of English to converse confidently and groups of Israelis often stick together in hostels. This was a common complaint of those that I befriended.
3) Not meeting any Irish people for the first 5 months until I was on a tour with a group of 6 of them.
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
    dragmode = False,
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
    dragmode = False,
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
Each country looks roughly proportional to the amount of time I spent in each, however, my experiences were rather different from country to country:
            
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


st.markdown("""
Given at the time I was a 27 year old male from the UK who was travelling solo and staying in hostels, this is perhaps unsurprising.
            """)

st.title("Age & Sex")

st.markdown("""
In general I found the average traveller to be of a similar age to myself but I did meet some older people. The boxplot below shows the distribution of age split by sex.
            """)

df_age_boxplot = df.copy()
df_age_boxplot['location_grouped'] = np.where(df_age_boxplot['location_met'].isin(['Accomodation','Tour']), df['location_met'], 'Other')

categories_ordered = ['Accomodation', 'Tour', 'Other']
def sns_stripplot(df, ordered = None, dodge_toggle = False):
    # Ensure you are creating a figure with plt.figure() and not calling plt as a function
    fig = plt.figure(figsize=(12, 6))
    # sns.stripplot is correctly used here as a function call
    sns.swarmplot(data=df, x="age", y="location_grouped", hue="sex", dodge = dodge_toggle, order = ordered)#, jitter = True)
    plt.xlabel("Age")
    plt.ylabel("Location Met")


    return fig

# Correct usage of the defined function to generate a figure
fig_boxplot = sns_stripplot(df=df_age_boxplot, ordered = categories_ordered)
st.pyplot(fig_boxplot)

df_age_boxplot_accom = df_age_boxplot[df_age_boxplot['location_met']=="Accomodation"]

st.markdown("""
Something that immediately catches my eye here is the number of entries at age = 27. This is likely because I did not ask everybody their age and often had to make an
educated guess. I suspect the question I asked myself at the time was 'Do I think they're about the same age as me?'.

While not as pronounced, a similar effect can be observed at ages of 5 year intervals i.e. 30, 40, 45. I almost certainly did not ask the ages of those above 50.

Something else that catches my eye is the strange clustering between males and females for those I met at my accommodation. The below
swarm plot focuses on just this category with a toggle to separate the sexes onto different lines
                        """)

dodge_output = st.toggle("Separate", False)
fig_boxplot_2 = sns_stripplot(df=df_age_boxplot_accom, dodge_toggle = dodge_output)
st.pyplot(fig_boxplot_2)

st.markdown("""
When the toggle is on, it looks as if we have two clusters for females, young twenties and older twenties, whereas males
tend to have a more normal distirbution centred around the mid-20s (with an additional peak at 30).
""")

options = ["Baby", "Septuagenarians"]
selection = st.radio("Outlier Choice", options)  # Using radio for a better example

# Create a figure for the swarm plot
figa, axa = plt.subplots(figsize=(12, 6))
sns.swarmplot(data=df, x="age", y="sex", ax=axa)

# Calculate y position for the circle based on the 'sex' category
circle_center = (0, 15)  # Set this to the desired center of the circle (x, y)
circle_radius = 3        # Set the desired radius of the circle
circle_color = 'red'     # Circle color

# Create a circle
circle = Circle(circle_center, circle_radius, color=circle_color, fill=False, linewidth=2)

# Get current axis and add circle to it
axa.add_patch(circle)


# Set labels
plt.xlabel("Age")
plt.ylabel("Sex")


st.pyplot(figa)

st.markdown("""
However, when we look at age distribution for both sexes across all locations met, it becomes more apparent that
females have clusters at certain ages. This could be either due to a small sample size, or more likely, my reluctance to ask ages and therefore guess.
Since I may have more information or at least more confidence in guessing the age of a male, I have a more even distribution of data.
            """)

st.title("Traveller Type")

st.markdown("""
Below is a bar chart of the different traveller types I categorised people into. Some of these are rather broad such as 'Resident' which encompasses those who lived, worked or studied in the country.
           """)

df_ttype = df.copy()
df_ttype['travelling_as'] = np.where(df_ttype['travelling_as'].isin(['Siblings']), 'Family', df_ttype['travelling_as'])
# size is useful to reduce to one column rather than having the same number of columns as the source df
df_ttype = df_ttype.groupby(["travelling_as"]).size().reset_index(name='count')

fig_ttype = plt.figure(figsize =(12,6))
sns.barplot(df_ttype, x="count", y = 'travelling_as', order = ["Family","Couple","Resident","Friends","Solo"])
plt.xlabel("Count")
plt.ylabel("Traveller Type")
st.pyplot(fig_ttype)

st.markdown("""
As a solo traveller you often go out of your way to meet other people. This is reflected in the data in both senses, I was looking for
people to socialise with and other solo travellers were seeking me out.

That being said, when travelling solo, you are approached far more often from those in groups/pairs
(and more receptive to being approached) than when travelling with others. Usually the leading question is 'Are you travelling alone?'.
""")

st.title("Meeting Method")


st.markdown("""
Getting into the slightly more obscure now, below is a bar plot of all the methods I met people.
""")

df_lmet = df.copy()
# size is useful to reduce to one column rather than having the same number of columns as the source df
df_lmet = df_lmet.groupby(["location_met"]).size().reset_index(name='count')
df_lmet = df_lmet.sort_values("count")

fig_lmet = plt.figure(figsize =(12,6))
sns.barplot(df_lmet, x="count", y = 'location_met')
plt.xlabel("Count")
plt.ylabel("Location Met")
st.pyplot(fig_lmet)

st.title("Outliers")


st.markdown("""

           """)

st.subheader("Appendix - Full Dataset")
with st.expander("Source Data"):
    df

