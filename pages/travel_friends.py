import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("data/data/people_names_redacted.csv")

st.title("Travel Friends")

with st.expander("Packages Used"):
    st.code("""
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
            """)

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
When plotting the nationalities of fellow travellers on a Choropleth Map, it can be seen that people I met tend to be from countries
that are well-developed economically and culturally similar to the UK.
            
Some surprises for me were 
1) Not meeting anybody from either Africa or Asia (not including Russia and Turkey).
2) The number of Israelis I met (9) especially considering many do not have a level of English to converse confidently and groups of Israelis often stick together in hostels. This was a common complaint of those that I befriended.
3) Not meeting any Irish people for the first 5 months until I was on a tour with a group of 6 of them.
            """)


# Aggregate data to count occurrences of each country
country_counts = df['country_origin'].value_counts().reset_index()
country_counts.columns = ['Country', 'Counts']

# Create choropleth figure
fig = px.choropleth(
    country_counts, 
    locations="Country",
    locationmode='country names',
    color="Counts",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Viridis, 
    labels={'Counts':'People Met'}
)

fig.update_geos(
    visible=False,  
    showcountries=True,  # Show country borders
    countrycolor="LightGrey"  # Set country border color to light grey
)

fig.update_layout(
    title_text='Nationality Count of Travellers I Met',
    title_x=0.4,
    dragmode = False,
    geo=dict(
        lakecolor='White', 
        showland=True,
        landcolor='White',
        showocean=True,
        oceancolor='LightBlue', 
        projection_type='equirectangular'
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="RebeccaPurple"
    )
)

st.plotly_chart(fig)
    
with st.expander("See Code"):
    st.code('''
# Aggregate data to count occurrences of each country
country_counts = df['country_origin'].value_counts().reset_index()
country_counts.columns = ['Country', 'Counts']

# Create choropleth figure
fig = px.choropleth(
    country_counts, 
    locations="Country",
    locationmode='country names',
    color="Counts",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Viridis, 
    labels={'Counts':'People Met'}
)

fig.update_geos(
    visible=False,  
    showcountries=True,  # Show country borders
    countrycolor="LightGrey"  # Set country border color to light grey
)

fig.update_layout(
    title_text='Nationality Count of Travellers I Met',
    title_x=0.4,
    dragmode = False,
    geo=dict(
        lakecolor='White', 
        showland=True,
        landcolor='White',
        showocean=True,
        oceancolor='LightBlue', 
        projection_type='equirectangular'
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="RebeccaPurple"
    )
)
    ''')

st.markdown("""
Below is a plot of the number of travellers I met in each country I visited. Each country looks roughly proportional to the amount of time I spent in that country,
however, when toggling to people met per day, it can be seen that I met almost twice as many people per day in Chile than the others.
            """)

# Aggregate data to count occurrences of each country
country_counts_met = df['country_met'].value_counts().reset_index()
country_counts_met.columns = ['Country', 'Counts']

# Second Choropleth plot
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
    visible=False,  
    showcountries=True,  
    countrycolor="LightGrey",
    scope='south america'  
)

fig_2.update_layout(
    title_text='Location of Meeting Place',
    title_x=0.4,
    dragmode = False,
    geo=dict(
        lakecolor='White',  
        showland=True,
        landcolor='White',
        showocean=True,
        oceancolor='LightBlue', 
        projection_type='equirectangular'
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="RebeccaPurple"
    )
)

# Aggregate data for a count of people met per country
df_country_duration = pd.read_csv("data/data/days_in_countries.csv")
df_country_duration = df_country_duration.groupby(["country"]).sum()
df_country_duration = df_country_duration["days_spent"].copy()
df_country_duration = df_country_duration.sort_values(ascending=False)

# Merge the two data sets
df_countries_merged = pd.merge(country_counts_met,df_country_duration,'inner', left_on = "Country", right_on = "country")

# Create new column for pm/pd
df_countries_merged["people_met_per_day"] = df_countries_merged["Counts"]/df_countries_merged["days_spent"]


fig_people_met_per_day = px.choropleth(
    df_countries_merged, 
    locations="Country",
    locationmode='country names',
    color="people_met_per_day",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Viridis, 
    labels={'Counts':'People Met Per Day'}
)

fig_people_met_per_day.update_geos(
    visible=False,  # Turn off the default geography visibility
    showcountries=True,  # Show country borders
    countrycolor="LightGrey",
    scope='south america'  
)

fig_people_met_per_day.update_layout(
    title_text='People Met Per Day',
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
    ),
    coloraxis_colorbar=dict(
        title=" "
    )
)

country_met_map_output = st.toggle("Per Day", False)

st.plotly_chart(fig_people_met_per_day if country_met_map_output else fig_2)

with st.expander("See Code"):
    st.code('''
# Aggregate data to count occurrences of each country
country_counts_met = df['country_met'].value_counts().reset_index()
country_counts_met.columns = ['Country', 'Counts']

# Second Choropleth plot
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
    visible=False,  
    showcountries=True,  
    countrycolor="LightGrey",
    scope='south america'  
)

fig_2.update_layout(
    title_text='Location of Meeting Place',
    title_x=0.4,
    dragmode = False,
    geo=dict(
        lakecolor='White',  
        showland=True,
        landcolor='White',
        showocean=True,
        oceancolor='LightBlue', 
        projection_type='equirectangular'
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="RebeccaPurple"
    )
)

# Aggregate data for a count of people met per country
df_country_duration = pd.read_csv("data/data/days_in_countries.csv")
df_country_duration = df_country_duration.groupby(["country"]).sum()
df_country_duration = df_country_duration["days_spent"].copy()
df_country_duration = df_country_duration.sort_values(ascending=False)

# Merge the two data sets
df_countries_merged = pd.merge(country_counts_met,df_country_duration,'inner', left_on = "Country", right_on = "country")

# Create new column for pm/pd
df_countries_merged["people_met_per_day"] = df_countries_merged["Counts"]/df_countries_merged["days_spent"]


fig_people_met_per_day = px.choropleth(
    df_countries_merged, 
    locations="Country",
    locationmode='country names',
    color="people_met_per_day",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Viridis, 
    labels={'Counts':'People Met Per Day'}
)

fig_people_met_per_day.update_geos(
    visible=False,  # Turn off the default geography visibility
    showcountries=True,  # Show country borders
    countrycolor="LightGrey",
    scope='south america'  
)

fig_people_met_per_day.update_layout(
    title_text='People Met Per Day',
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
    ),
    coloraxis_colorbar=dict(
        title=" "
    )
)

# Toggle to change to per day view.
country_met_map_output = st.toggle("Per Day", False)

# Default behaviour is to show totals, per day will be passed when toggle is on
st.plotly_chart(fig_people_met_per_day if country_met_map_output else fig_2)
            ''')

st.markdown("""
It is hard to specify when this might be as my experiences were rather different from country to country:
            
<ul>
            <li> In <b>Peru</b> my main activities were Spanish School and the Salkantay trek, both of which meant spending considerable time with the same people.
            <li> In <b>Chile</b> I was primarily hiking and camping solo, this likely prompted me to be more sociable.
                <ul>
                    <li> A lot of the people I met in Chile I then met again later on in the trip. This means they are attributed only once as met in Chile.
                </ul>
            <li> <b>Argentina</b> was the country I spent by far the most time in and through my girlfriend met many locals.
            <li> I spent the least time in <b>Bolivia</b> but did take another two weeks of Spanish Lessons which invariably leads to making connections.
                <ul>
                    <li> Bolivia was also the last country I visited and the country where I met a lot of the same people I had earlier in the trip.
                    <li> Spending time with people I previously met meant I was putting less effort into meeting new people.
                </ul>
</ul>
            """, unsafe_allow_html=True)


st.title("Age & Sex")

st.markdown("""
In general I found the average traveller to be of a similar age to myself but I did meet some older people. The boxplot below shows the distribution of age split by sex.
            """)

# Copy of original data frame
df_age_boxplot = df.copy()
# Similar to a case when statement, if the location met is in two main categories then remain the same, bucket everything else into other
df_age_boxplot['location_grouped'] = np.where(df_age_boxplot['location_met'].isin(['Accomodation','Tour']), df['location_met'], 'Other')

# List created for ordering plot
categories_ordered = ['Accomodation', 'Tour', 'Other']

# Creating functions for every plot allows for lazy naming conventions wrt 'fig' etc. Review whether this is appropriate
def sns_stripplot(df, ordered = None, dodge_toggle = False):
    fig = plt.figure(figsize=(12, 6))
    sns.swarmplot(data=df, x="age", y="location_grouped", hue="sex", dodge = dodge_toggle, order = ordered)#, jitter = True)
    plt.xlabel("Age")
    plt.ylabel("Location Met")
    return fig

# Now assign a variable name to the complete plot
fig_boxplot = sns_stripplot(df=df_age_boxplot, ordered = categories_ordered)
st.pyplot(fig_boxplot)

with st.expander("See Code"):
    st.code('''
# Copy of original data frame
df_age_boxplot = df.copy()
# Similar to a case when statement, if the location met is in two main categories then remain the same, bucket everything else into other
df_age_boxplot['location_grouped'] = np.where(df_age_boxplot['location_met'].isin(['Accomodation','Tour']), df['location_met'], 'Other')

# List created for ordering plot
categories_ordered = ['Accomodation', 'Tour', 'Other']

# Creating functions for every plot allows for lazy naming conventions wrt 'fig' etc. Review whether this is appropriate
def sns_stripplot(df, ordered = None, dodge_toggle = False):
    fig = plt.figure(figsize=(12, 6))
    sns.swarmplot(data=df, x="age", y="location_grouped", hue="sex", dodge = dodge_toggle, order = ordered)#, jitter = True)
    plt.xlabel("Age")
    plt.ylabel("Location Met")
    return fig

# Now assign a variable name to the complete plot
fig_boxplot = sns_stripplot(df=df_age_boxplot, ordered = categories_ordered)
st.pyplot(fig_boxplot)
    ''')

st.markdown("""
Something that immediately catches my eye here is the number of entries at age = 27. This is likely because I did not ask everybody their age and often had to make an
educated guess. I suspect the question I asked myself at the time was 'Do I think they're about the same age as me?'.

While not as pronounced, a similar effect can be observed at ages of 5 year intervals i.e. 30, 40, 45. I almost certainly did not ask the ages of those above 50.

Something else that catches my eye is the strange clustering between males and females for those I met at my accommodation. The below
swarm plot focuses on just this category with a toggle to separate the sexes onto different lines
                        """)

# Create toggle for splitting by sex
# this works because in the function we set the hue to sex and added an optional dodge toggle argument
dodge_output = st.toggle("Separate", False)

# Filter just for those met in accommodation
df_age_boxplot_accom = df_age_boxplot[df_age_boxplot['location_met']=="Accomodation"]
fig_boxplot_2 = sns_stripplot(df=df_age_boxplot_accom, dodge_toggle = dodge_output)
st.pyplot(fig_boxplot_2)

with st.expander("See Code"):
    st.code("""
# Create toggle for splitting by sex
# this works because in the function we set the hue to sex and added an optional dodge toggle argument
dodge_output = st.toggle("Separate", False)

# Filter just for those met in accommodation
df_age_boxplot_accom = df_age_boxplot[df_age_boxplot['location_met']=="Accomodation"]
fig_boxplot_2 = sns_stripplot(df=df_age_boxplot_accom, dodge_toggle = dodge_output)
st.pyplot(fig_boxplot_2)
""")

st.markdown("""
When the toggle is on, it looks as if we have two clusters for females, young twenties and older twenties, whereas males
tend to have a more normal distirbution centred around the mid-20s (with an additional peak at 30).
""")

options = ["Baby", "Golf", "Retiree"]
selection = st.pills("Outlier Explainer", options, selection_mode="single")

figa, axa = plt.subplots(figsize=(12, 6))
sns.swarmplot(data=df, x="age", y="sex", ax=axa)

if selection == "Baby":
    axa.annotate(
        "Baby of German couple on joint mat/pat leave \n met when they picked me up at the side of the road.",
        xy=(0, 0),  
        xycoords='data',
        xytext=(-10, -90), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    )

if selection == "Golf":
     axa.annotate(
        "Randomly paired with playing \n golf in Buenos Aires.",
        xy=(75, 1),  # Point to F on the y-axis and age 0
        xycoords='data',
        xytext=(-250, 80),  
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    )
     axa.annotate(
        "",
        xy=(73, 1),  
        xycoords='data',
        xytext=(-130, 70), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    ) 
     axa.annotate(
        "",
        xy=(59, 1), 
        xycoords='data',
        xytext=(-40, 75),  
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    ) 

if selection == "Retiree":
    axa.annotate(
        "Recently retired Belgian lady on first \nsolo trip, had always wanted to hike \nin Patagonia but her husband had no \ninterest.",
        xy=(68, 0),  
        xycoords='data',
        xytext=(-150, -90), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    )

axa.set_xlabel("Age")
axa.set_ylabel("Sex")

st.pyplot(figa)

with st.expander("See Code"):
    st.code('''
options = ["Baby", "Golf", "Retiree"]
selection = st.pills("Outlier Explainer", options, selection_mode="single")

figa, axa = plt.subplots(figsize=(12, 6))
sns.swarmplot(data=df, x="age", y="sex", ax=axa)

if selection == "Baby":
    axa.annotate(
        "Baby of German couple on joint mat/pat leave \n met when they picked me up at the side of the road.",
        xy=(0, 0),  
        xycoords='data',
        xytext=(-10, -90), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    )

if selection == "Golf":
     axa.annotate(
        "Randomly paired with playing \n golf in Buenos Aires.",
        xy=(75, 1),  # Point to F on the y-axis and age 0
        xycoords='data',
        xytext=(-250, 80),  
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    )
     axa.annotate(
        "",
        xy=(73, 1),  
        xycoords='data',
        xytext=(-130, 70), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    ) 
     axa.annotate(
        "",
        xy=(59, 1), 
        xycoords='data',
        xytext=(-40, 75),  
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    ) 

if selection == "Retiree":
    axa.annotate(
        "Recently retired Belgian lady on first \nsolo trip, had always wanted to hike \nin Patagonia but her husband had no \ninterest.",
        xy=(68, 0),  
        xycoords='data',
        xytext=(-150, -90), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5),
        fontsize=10,
        color='red'
    )

axa.set_xlabel("Age")
axa.set_ylabel("Sex")

st.pyplot(figa)

''')

st.markdown("""
However, when we look at age distribution for both sexes across all locations met, it becomes more apparent that
females have clusters at certain ages. This could be either due to a small sample size, or more likely, my reluctance to ask ages and therefore guess.
Since I may have more information or at least more confidence in guessing the age of a male, I have a more even distribution of data.
            """)

st.title("Traveller Type")

st.markdown("""
Below is a bar chart of the different traveller types I categorised people into. Some of these are rather broad such as 'Resident' which encompasses those who were living, working or studying in the country.
           """)

# Create a copy of the original df
df_ttype = df.copy()

# Case statement to group siblings into a family category
df_ttype['travelling_as'] = np.where(df_ttype['travelling_as'].isin(['Siblings']), 'Family', df_ttype['travelling_as'])
# size is useful to reduce to one column rather than having the same number of columns as the source df
df_ttype = df_ttype.groupby(["travelling_as"]).size().reset_index(name='count')

fig_ttype = plt.figure(figsize =(12,6))
sns.barplot(df_ttype, x="count", y = 'travelling_as', order = ["Family","Couple","Resident","Friends","Solo"])
plt.xlabel("Count")
plt.ylabel("Traveller Type")
st.pyplot(fig_ttype)

with st.expander("See Code"):
    st.code("""
# Create a copy of the original df
df_ttype = df.copy()

# Case statement to group siblings into a family category
df_ttype['travelling_as'] = np.where(df_ttype['travelling_as'].isin(['Siblings']), 'Family', df_ttype['travelling_as'])
# size is useful to reduce to one column rather than having the same number of columns as the source df
df_ttype = df_ttype.groupby(["travelling_as"]).size().reset_index(name='count')

fig_ttype = plt.figure(figsize =(12,6))
sns.barplot(df_ttype, x="count", y = 'travelling_as', order = ["Family","Couple","Resident","Friends","Solo"])
plt.xlabel("Count")
plt.ylabel("Traveller Type")
st.pyplot(fig_ttype)
            """)

st.markdown("""
That being said, when traveling solo, you are approached far more often by those in groups or pairs 
(and are more receptive to being approached) than when traveling with others.
 Usually, the leading question is, 'Are you traveling alone?'
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

st.markdown("""
Some of these are self-explanatory but I will explain those that are more interesting than others.
            
#### Wallyball

<a href="https://www.youtube.com/watch?v=88iHgCwVzV0&ab_channel=BRICTV">Wallyball</a> is a variant of Volleyball played inside where the width and length of the normal size volleyball pitch
are contained by walls. You are allowed to use any part of your body and the ball is always in play as long as it doesn't hit
the back wall or floor before any other surface.

We played as part of my Spanish School in Sucre, Bolivia and somebody brought along a friend who I got to know better later on.
            
""", unsafe_allow_html=True)

with st.expander("See Wallyball"):
    st.image("media/ezgif-5d6b19265fa17a.gif", caption="Winning a point in wallyball with my foot.")

st.markdown("""

#### Thermal Baths

After some long trekking days and while staying in a particularly dull town in Chilean Patagonia, I decided to treat myself to a trip to the local thermal bath. I decided to run to the venue but it was longer than I anticipated.
Fortunately I met a lovely Belgian couple in one of the pools who gave me a lift back to town in their campervan. I would go on to meet them again by chance in Sucre, Bolivia.


#### Hitchhiking

I depended a lot on hitchhiking while exploring Patagonia. Not only are buses fairly infrequent but they often don't stop exactly where you want them to and they can work out very expensive.
Hitchhiking has the benefit of being free but also providing a good opportunity to practice language skills and make friends. Obviously I recognise my privilige as a male, but I strongly recommend it as a means of transport
in this part of the world. 
            
That being said, you are never guaranteed a lift and for some of the harder to reach destinations, I felt it best to rent my own car for 4 days. During this time it was only right that I returned the favour of picking up hitchhikers
I saw along the way.
            
One of the passengers I picked up in a remote part of Chilean Patagonia, I later met up with for lunch at a food market in Sucre, Bolivia.
                    
#### Girlfriend

I met my girlfriend the first time I went to Buenos Aires in March 2023. I then returned to BA on two more occasions and through her was able to meet far more 'real' Argentinians than I would have been able to otherwise. Out of respect for her
privacy I'll resist the temptation to attach an accompanying large embarrassing photo.


            
""", unsafe_allow_html=True)


st.subheader("Appendix - Full Dataset")
with st.expander("Source Data"):
    df

