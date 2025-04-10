
import streamlit as st

st.title("Deploying This Website to Google Cloud")

st.header("Introduction", divider = True)

st.write(
"""
Thus far I have been using a Github codespace to develop this website and hosting for free on streamlit community cloud.
I like this approach because any changes I make are automatically pushed to production when I make commits in the IDE.

However, there are two drawbacks with hosting on community cloud. First, the app will go to sleep after 12 hours of inactivity.
This is not too much of an issue, however, given I am using this site to appeal to potential employers I would like to appear as 
professional as possible. I don't want people to have to 'wake' my site up and wait a couple of minutes for the necessary packages to install.
"""
)

st.image("media/app-state-zzzz.png")
st.caption("Not the best user experience...")

st.write(
"""
Secondly, the url of https://hallam-flda.streamlit.app/ is not as clean as I would like. For this reason I decided to also host this page on my own domain using
[Google Cloud Run](https://cloud.google.com/run?_gl=1*1f6zcgi*_up*MQ..&gclid=CjwKCAjwtdi_BhACEiwA97y8BI0qFqNhp2SVTIr16az3hagz1ts_TsjS8uHLf6xGD6aGSFT5vr0fBxoCYPUQAvD_BwE&gclsrc=aw.ds&hl=en).
"""
)

st.header("Hosting on Google Cloud Run", divider = True)

st.subheader("Creating a Container/Dockerfile")
st.write("""
In order to port the working web app to Google, I need to put it in a container. Docker is the most common tool for doing this
         
The first step is to create a Dockerfile in the root folder of my project. What is a Dockerfile? From the Docker website:
  """
)

st.markdown("> *“A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings.”*  \n> ", unsafe_allow_html=True)

st.write(
"""
In other words, a way to move software across different computers and operating systems. The Dockerfile looks like this:
""")

st.code(
"""
# Define which language the programme is written in
FROM python:3.11

# Set a working directory for the container to use
WORKDIR /app

# Install existing dependencies from the root folder into the newly created working directory
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the project over
COPY . .

# Run the app in the same way as usual in the command line
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
"""
, language = 'docker')

st.write(
"""
Now that the dockerfile is ready to go, it just needs to be deployed somewhere.
"""
)

st.subheader("Deploying to Google Cloud")

st.write(
"""
Now direct from the terminal within my github codespace I am able to push the containerized app to Google Cloud Run. First authorise through the gcloud CLI tool
"""
)

st.code("""
gcloud auth login
gcloud config set project tonal-run-447413-c0
""", language = 'bash'
)

st.write(
"""
This will prompt you to login in using google credentials.

Then the command to build the app can be run
"""    
)

st.code("gcloud builds submit --tag gcr.io/tonal-run-447413-c0/streamlit-app", language = 'bash')

st.write(
"""
This creates a cloud run app called streamlit-app but the app isn't deployed until some more settings are defined.
"""
)

st.code(
"""
gcloud run deploy streamlit-app \

  --image gcr.io/tonal-run-447413-c0/streamlit-app \
  
  --platform managed \
  
  --region europe-west1 \
  
  --allow-unauthenticated
"""
, language='bash')

st.write(
"""
The app is now deployed and live! That is the first aim of hosting on another platform complete. However, the default link: https://streamlit-app-721866480619.europe-west2.run.app is rather unsightly.

Fortunately, cloud run allows you to use a custom domain to reach the hosted web-app. This means I need to buy a domain.
"""
)

st.header("Purchasing the Domain", divider = True)

st.write(
"""
This was far easier than I anticipated, all I needed to do was search for the URL I wanted on Squarespace and click buy. The logical 
choice here was my full name, even if this site is only used short-term, £10 (or 67 Brazilian Reais) a year for my own URL felt like a good investment.
"""    
)

st.image("media/url_options.png")
st.caption("My sincere apology to any other Hallam Cunninghams but you can have the .info for a great price")

st.write(
"""
Now all I need to do is map the domain that I own from Squarespace to the Google Cloud app I have hosted in GCP.
"""
)

st.header("Using my Custom Domain", divider = True)

st.write(
"""
In Google Cloud Run, there is an option to manage and map custom domains.
"""
)

st.image("media/domain_mapping.png")

st.write(
"""
As you can see I have already completed the process and truthfully this is a bit beyond my understanding so I won't post all the details for fear of oversharing. In short the process involves
generating some IP addresses that people will be directed to when they use my domain https://hallamcunningham.com. These IP addresses are owned by Google and point to the deployed app I created earlier.
"""
)

st.header("Next Steps", divider=True)

st.write(
"""
Now the website is live at my custom URL, however, I need to work out how to also map www.hallamcunningham.com because despite the fact I own that, it is another URL that needs to be mapped to the Cloud Run app.
"""
)