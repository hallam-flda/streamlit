import streamlit as st
import streamlit.components.v1 as components

st.title("Dbt Fundamentals")
st.caption("Full project is available to view on [Github](https://github.com/hallam-flda/garmin-dbt) ")

st.header("Introduction", divider = True)
col1, col2 = st.columns([1,1])

with col1:

    
    st.write(
    """
    Increasingly it is becoming a requirement of Data Analysts to have an understanding of the end-to-end data process. In large organisations, analysts are usually served a final production quality
    data table from which they can serve business insights to stakeholders. I have found bridging the gap between engineers and stakeholders can be frustrating at times, especially when, as an analyst, you don't
    fully understand the ETL/ELT process yourself.

    For this reason, I have followed Dbt's extremely useful [fundamentals](https://learn.getdbt.com/learn/course/dbt-fundamentals/) course to create a job based off my [garmin map](https://hallam-flda.streamlit.app/garmin_activity_map) workflow.
    This taught me a lot on how data should be structured and the difference between development/testing/production environments, as well as the need for staging tables etc. I would recommend any 
    analyst who does not have a good understanding of data engineering to take the course, even if they never need to use Dbt.
    """    
    )

with col2:
    
    # # URL of the image
    # image_url = "https://api.accredible.com/v1/frontend/credential_website_embed_image/badge/138653777"

    # # Display the image
    # st.image(image_url, caption="A nice shiny badge! \n of course this could just be a saved image \n but if you copy the link you'll see it \n comes from accredible's API (promise!)", use_container_width=True)
    
    components.iframe(
        "https://credentials.getdbt.com/embed/dff728a0-039e-44c6-845a-c3c538ac99b6",
        width=350,
        height=500,
        scrolling=True
    )
    
    st.caption("Proof I got my accreditation!")
    
st.header("The Use Case - Garmin Data", divider = True)

st.write(
"""
The course provides a good set of example data, however, I find that the knowledge sticks better when I'm working on something I care about. When working on my garmin map visualisation, I did a lot of 
data cleaning, most of which is documented in the write up, however, it certainly was not best practise, nor did I organise the files efficiently enough to be able to reproduce easily with new data. What I needed was
a fully production-ready data workflow to take raw data and output data in the same format as kepler takes in the kepler UI.

Jumping ahead slightly, the final DAG (directed acyclic graph) looks like this:
"""    
)

st.image("media/garmin/dbt-dag.png")

st.write(
"""
I will guide through what happens at each stage and add some general commentary about the process.
"""    
)

st.header("Source", divider = True)

st.write(
"""
In Dbt we can designated sources by defining them in a YAML file within our staging folder. There is no model within the Dbt ecosystem per se, rather it plugs in directly to 
the database following the path defined in the YAML file. In this instance the file is named `_garmin__sources.yml`

This is my first time using YAML files as well, but the syntax is fairly straightforward, I think of it as a .txt file but with some structure. This file is pointing at `garmin.activites` in my
BigQuery environment as determined by the `schema:` and `name:` fields.
"""    
)

with st.expander("See YAML source file"):
    st.code(
    """
    version: 2

    sources:
    - name: garmin
        description: Export using fitfiles from .fit files sent by GarminConnect
        database: tonal-run-447413-c0     # BigQuery project name
        schema: garmin               # BigQuery dataset name
        tables:
        - name: activities         # BigQuery table name
            description: "Raw activity data from Garmin"
            # WILL BE USEFUL LATER
            # loaded_at_field: _etl_loaded_at
            # freshness:
            #   warn_after: {count: 3, period: day}
            columns:
            - name: timestamp
                tests:
                - unique
                - not_null
                description: primary key for export of .fit files
            - name: position_lat
                tests:
                - not_null:
                    severity: warn
            - name: position_long
                tests:
                - not_null:
                    severity: warn
    """  
    , language = "yaml")

st.write(
"""
Also defined in this file are some tests we can do to ensure the completeness of our data and make sure that it won't cause any issues downstream. I'll get into more detail about this later, however,
as you can see, I don't currently have a good unique key, nor a column to test data freshness. These would both be required in a production grade pipeline.
"""    
)

st.header("Staging", divider = True)

st.write(
"""
Prior to starting the course, I had some awareness of what staging was but not really the purpose it served. As I understand it now, staging data should be as close to the source data as possible, the
only changes would be some very light transformations such as converting units or casting to different data types. 

Staging is the first defence against changing data in your source table. If something unexpected happens with the formatting of the source data, then it should be caught with appropriate testing and casting within
the stage step.

In my case, all I am doing is rounding some columns that have a large number of trailing decimal places and removing any null entries for geolocation data as I cannot plot this.
"""    
)

with st.expander("See Staging SQL"):
    st.code(
        """
        with staging_query as (
        select
            cadence,
            distance as distance_meters,
            ROUND(distance/1000,2) distance_km,
            ROUND(enhanced_altitude,0) as altitude,
            ROUND(enhanced_speed,2) as speed,
            heart_rate,
            position_lat * (180/2147483648) position_lat,
            position_long * (180/2147483648) position_long,
            CAST(timestamp AS TIMESTAMP) timestamp
        from 
            {{ source('garmin', 'activities') }}

        WHERE
            position_lat is not null and position_long is not null
            )

        select * from staging_query
        """, language = "sql")

st.write(
"""
A key part of what makes Dbt so easy to use is the macros. I did not create the DAG, nor did I put any effort into writing the documentation, this has all been tracked through the use of macros. In
this example rather than just write the name of the source table, I have used `{{ source('garmin', 'activities') }}` and Dbt now knows to look at the source defined in the YAML file.
"""    
)

st.header("Intermediate", divider = True)

st.write(
"""
This did not form part of the tutorial but after deducing that my original SQL script did more than just some light formatting, I determined that I should create an intermediatery step in the process. This is where I
create session flags by lagging the timestamp and taking differences greater than 6 hours to be a new session.

Again the code uses a reference macro to the staging model which links this script as part of a downstream process, one that can only begin if all the previous steps have completed without error.
"""    
)

with st.expander("See Intermediate SQL"):
    st.code(
        """

        with activities as 
        (
            select
            *,
            lag(timestamp,1) over (order by timestamp asc) lag_timestamp

            from
            {{ ref('stg_garmin_activities') }}

        )

        , flag_large_time_difference as 
        (
            SELECT
            *,
            TIMESTAMP_DIFF(timestamp, lag_timestamp, HOUR) time_diff_hours,
            CASE 
            WHEN TIMESTAMP_DIFF(timestamp, lag_timestamp, HOUR) >= 6 THEN 1 
            ELSE 0 
            END AS activity_end_flag
        FROM activities
        )

        ,session_flag as 
        (
        SELECT 
        *,
        sum(activity_end_flag) over (order by timestamp asc) activity_id
        FROM flag_large_time_difference 
        )

        select
        activity_id,
        cadence,
        distance_meters,
        distance_km,
        altitude,
        speed,
        heart_rate,
        position_lat,
        position_long,
        timestamp

        from session_flag

        order by timestamp asc
        """
        , language = 'sql')

st.write(
"""
This creates the equivalent of `garmin.activities_clean` dataset in BigQuery. However, now the tables are being written to a new schema of `dbt_hcunningham` which was defined in the set-up of this 
project. Consequently, this table is now called `dbt_hcunningham.int_activities_clean`.
"""    
)

st.header("Marts", divider = True)

st.write(
"""
Marts are the final products, these are what is served to the teams who need to access data and they are served usually in one of two formats: Dimension Tables and Fact Tables.
"""    
)

st.subheader("Dimension Tables")
st.write(
"""
Dimension Tables are those that are characterised by their primary key and are likely to have some attributes that give some more context to the primary key. An example of this would be a customer ID with all
additional columns providing additional context about that customer. These fields can also be expected to change from time to time, say if a customer changes their email address or even one of their names.
"""
)

st.subheader("Fact Tables")
st.write(
"""
Fact tables are logs of recorded factual activity. Often they will be narrow and long tables and you would not expect any of the fields to change as the data recorded should be preserved in its moment in time.

For this reason, all of my output tables are fact tables. Every row represents a snapshot of my location and health stats at a moment in time and these entries will never change.
"""
)

st.subheader("Activity Logs")
st.write(
"""
The first of my fact tables is the activity log, this provides all the data required to plot my movements on the map in 1-3 second intervals. The initial run of this code produces a table with 22 columns and 190,000 rows.

As discussed in the original write-up, this is not all completely necessary, some of the columns are nice-to-have and the frequency of the geolocation is a bit overkill. To counter this I have created
an additional fact table that is a condensed version of the first.
"""
)

with st.expander("See Activity Log SQL"):
    st.code(
        """
        WITH
        base_table AS (
        SELECT
            *,
            MAX(timestamp) OVER (PARTITION BY activity_id) end_time,
            MIN(timestamp) OVER (PARTITION BY activity_id) start_time,
            LEAD(timestamp) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_timestamp,
            LEAD(position_lat) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_lat,
            LEAD(position_long) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_long,
            MAX(distance_meters) OVER (PARTITION BY activity_id) end_distance_meters,
            MIN(distance_meters) OVER (PARTITION BY activity_id) start_distance_meters
        FROM
            {{ ref('int_activities_clean') }}
            )

        , distance_and_time_diffs as 
        (  
        SELECT
        *,
        TIMESTAMP_DIFF(lead_timestamp, timestamp, SECOND) time_increment,
        -- to remove the row where activities lapse
        ST_DISTANCE( ST_GEOGPOINT(position_long, position_lat), ST_GEOGPOINT(lead_long, lead_lat) ) / 1000 dist_to_next_row_km
        FROM
        base_table
        )

        , activity_time as 
        (
            SELECT
            * EXCEPT (time_increment),
            sum(time_increment) over (partition by activity_id) activity_duration_seconds

            from
            distance_and_time_diffs

            where 
            dist_to_next_row_km < 1

        )

        select
        *,
        ROUND(activity_duration_seconds/60,2) activity_duration_minutes,
        ROUND(end_distance_meters/activity_duration_seconds,2) avg_speed_ms,
        ROUND(60 / ((end_distance_meters/activity_duration_seconds)*3.6),2) pace_mins_per_km

        from
        activity_time

        order by activity_id asc, timestamp asc

        """
        , language='sql')
with st.expander("See Activity Log Condensed SQL"):
    st.code(
        """
        with
    base_table as (
        select
            *,
            rank() over (partition by activity_id order by timestamp asc) activity_step
        from {{ ref("fct_activity_log") }}
        order by activity_id asc, timestamp asc
    ),
    percentage_ranking as (
        select
            *,
            percent_rank() over (
                partition by activity_id order by activity_step
            ) percent_step
        from base_table
    ),
    flag_keep_rows as (
        select
            *,
            case
                when percent_step <= 0.05 or percent_step >= 0.95
                then 'Delete'
                when mod(activity_step, 5) = 0
                then 'Keep'
                else 'Delete'
            end
            keep_criteria
        from percentage_ranking
    ),
    condensed_rows as (
        select * except (keep_criteria, lead_lat, lead_long) from flag_keep_rows where keep_criteria = 'Keep'
    )

    -- have to reassess any large gaps between activities.

    ,lead_coords as (
        select
            *,
            max(timestamp) over (partition by activity_id) max_timestamp,
            lead(position_lat) over (
                partition by activity_id order by timestamp asc
            ) lead_lat,
            lead(position_long) over (
                partition by activity_id order by timestamp asc
            ) lead_long
        from condensed_rows
    )
    select
        activity_id,
        heart_rate,
        cadence,
        altitude,
        distance_km,
        position_lat,
        position_long,
        round(end_distance_meters/1000,2) end_distance_km,
        timestamp,
        activity_duration_seconds,
        activity_duration_minutes,
        avg_speed_ms,
        pace_mins_per_km,
        max_timestamp,
        lead_lat,
        lead_long,
        st_distance(
            st_geogpoint(position_long, position_lat), st_geogpoint(lead_long, lead_lat)
        )
        / 1000 dist_to_next_act

    from lead_coords

    where
        -- drop rows that jump large distances, this happens when either I forget to
        -- record or GPS is lost but looks strange on map.
        st_distance(
            st_geogpoint(position_long, position_lat), st_geogpoint(lead_long, lead_lat)/ 1000 < 1
        """
        , language='sql')
    
st.subheader("Travel Log")
st.write(
"""
The final fact table is simply a log of all the significant jumps in location (more than 30km) between activities, keeping only the columns necessary to plot start and end points. This will then need
to be passed to the google routes API to complete a full update of the map but that falls outside the scope of what Dbt can do.
"""
)

with st.expander("See Travel Log SQL"):
    st.code(
        """
        WITH
        base_table AS (
            SELECT
            *,
            LEAD(timestamp) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_timestamp,
            LEAD(position_lat) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_lat,
            LEAD(position_long) OVER (PARTITION BY activity_id ORDER BY timestamp ASC) lead_long
            FROM {{ ref('int_activities_clean') }}
        ),

        distance_and_time_diffs AS (
            SELECT
            *,
            ST_DISTANCE(
                ST_GEOGPOINT(position_long, position_lat),
                ST_GEOGPOINT(lead_long, lead_lat)
            ) / 1000 AS dist_to_next_row_km
            FROM base_table
        )

        SELECT 
        activity_id,
        position_lat,
        position_long,
        lead_lat,
        lead_long

        FROM distance_and_time_diffs
        WHERE dist_to_next_row_km > 30

        """, language = 'sql')

st.header("The End Product", divider = True)

st.write(
"""
As a result of this process, after loading the data in the source table `garmin.activities` all I need to do is run this job to populate all of the necessary downstream tables.

In the image below, we can see my source table still sits in the garmin schema but all of the others are now populated in the dbt_hcunningham schema. Furthermore, the staging and intermediate tables are
stored as views rather than tables. This means when queried, they reference the source table with the
query rather than being stored and will only run if they are queried directly (which there is little reason to do in this use-case.)
"""
)

st.image("media/garmin/end_dbt_view.png")

st.header("Other Useful Learnings", divider = True)

st.subheader("Tests")

st.write(
"""
One of the benefits of using Dbt is setting up tests on each model to ensure the final data appears in a clean and correct format.

When setting up any workflow there is no guarantee that the data we will receive in future will be the same as the initial run.
For this reason it is important to set up tests.

There are two types of test, generic and singular. I will define each briefly below.
"""
)

st.subheader("Generic Tests")

st.write(
"""
Generic tests are the standard, out-the-box, tests that we can apply to any column. These are: `unique`, `not_null`, `accepted_values` and `relationships`

Each of these tests generate an additional step in the DAG where we can only move to the next step if the test is passed. We could construct
these tests ourselves as they are simply additional SQL on a `select *` of the final model, but defining in a yaml file saves the headache of doing so.

* `unique`: checks that all the values in the column are unique, useful for primary keys and dimension tables.
* `not_null`: useful for any column that will be joined onto, therefore useful for foreign and primary keys.
* `accepted_values`: expects a list of values that are authorised to be in the table, good for spotting any new levels in a categorical column.
* `relationships`: checks that all the values of one column exist in another table/model referenced in the macro supplied. Useful in case of clashes in source data syncing.

I have not used tests too heavily in the garmin workflow, mainly because I already filter out the nulls and there is only one source table
so the chances of something going wrong as minimal, however, it's good to have an understanding of why I might need these in future.
"""    
)

st.subheader("Singular Tests")

st.write(
"""
Outside of the generic tests, we may want to test for some behaviour that is unique to the model in question. 

Here I could actually make good use of the tests, such as when the difference between any two rows in an acitvity is greater than 60 seconds, this would suggest
I had lost satellite data and therefore the activity data may not be reliable.
"""
)