import streamlit as st

st.title("Habit Tracking Dashboard")
st.caption("An Embedded Looker Studio Dashboard to track my daily habits. See expanders at bottom for workflows.")

# Embed Looker dashboard (Replace with your Looker URL)
looker_dashboard_url = "https://lookerstudio.google.com/embed/reporting/737ea117-2e83-45ed-8a2d-b506331e0ca1/page/6DAAF"

st.components.v1.iframe(looker_dashboard_url, width=800, height=600)

with st.expander("Duolingo Workflow"):
    st.header("Duolingo Daily Progress")
    st.markdown(
    """
    A python script hosted and scheduled on GCP runs every 4 hours to extract my
    XP values from the duolingo API. This then writes to a Google Sheet which is used as a 
    data source in Looker Studio. 
    
    Because of changes made to the duolingo API you can no longer
    access this information with login details, instead you have to provide a session cookie as if you
    were logged in through your browser. Details: [API](https://github.com/KartikTalwar/duolingo?tab=readme-ov-file) and fix for
    login details not working [here](https://github.com/KartikTalwar/Duolingo/issues/128).

    This could be improved by writing directly to a database in GCP but will suffice for now. I decided against coding directly in
    streamlit to avoid revealing my API details (not that they're particularly sensitive). I've since found out you can
    use secrets in streamlit so this was not an issue anyway.
    """)
    
    st.subheader("Python Code")
    st.code(
        """
        import duolingo
        import datetime
        import pandas as pd
        import json
        import inspect
        import numpy as np
        import gspread
        from google.cloud import secretmanager


        def google_sheets_function(request):

            gcp_creds_json = {
        "type": "service_account",
        "project_id": "tonal-run-447413-c0",
        "private_key_id": PRIVATE_KEY_ID,
        "private_key": PRIVATE_KEY,
        "client_email": CLIENT_EMAIL,
        "client_id": CLIENT_ID,
        "auth_uri": AUTH_URI,
        "token_uri": TOKEN_URI,
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": CLIENT_X509_CERT_URL,
        "universe_domain": "googleapis.com"
        }
            print(gcp_creds_json)

            source = inspect.getsource(duolingo)
            new_source = source.replace('jwt=None', 'jwt')
            new_source = source.replace('self.jwt = None', ' ')
            exec(new_source, duolingo.__dict__)
            
            lingo = duolingo.Duolingo(USER, jwt=JWT)
            api_lang_info = lingo.user_data.languages

            # Filter for Spanish and Portuguese points
            xp_data = {lang['language_string'].lower(): lang['points']
                    for lang in api_lang_info if lang['language_string'] in ['Spanish', 'Portuguese']}

            db_dict = {
                "datetime": datetime.datetime.now().isoformat(),
                "xp": xp_data
            }

            db_df2 = pd.DataFrame([db_dict])
            db_df_final = pd.concat([db_df2.drop(['xp'], axis=1), db_df2['xp'].apply(pd.Series)], axis=1)
            row_to_append = db_df_final.iloc[0].astype(object).to_list()
            row_to_append = [int(x) if isinstance(x, np.int64) else x for x in row_to_append]

            # Initialise Google Sheets client
            gc = gspread.service_account_from_dict(gcp_creds_json)
            sheet_url = '1q1xDAhCBlIs9VBPu698HSCysR72GWLJRu_0EBogYTgk'  # Google Sheet key
            sh = gc.open_by_key(sheet_url)
            worksheet = sh.sheet1
            worksheet.append_row(row_to_append)

            return "Row appended successfully"

        
        """
            )
