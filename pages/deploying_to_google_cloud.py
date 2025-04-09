
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
st.subheader("Creating a Dockerfile")
st.write("""

The first step is to create a Dockerfile in the root folder of my project. What is a Dockerfile? From the [dockerdocs](https://docs.docker.com/reference/dockerfile/):
`A Dockerfile is a text document that contains all the commands a user could call on the command line to assemble an image.`

Sadly this means nothing to me, so off to Mr GPT for an easier to understand definition

         """)



st.header("Purchasing the Domain", divider = True)

st.write(
"""
This was far easier than I anticipated, all I needed to do was search for the URL I wanted on Squarespace and click buy. The logical 
choice here was my full name, even if this site is only used short-term, £10 (or 67 Brazilian Reais) a year for my own URL felt like a good investment.
"""    
)

st.image("media/url_options.png")
st.caption("My sincere apology to any other Hallam Cunninghams but you can have the .info for a great price")

