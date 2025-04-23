FROM python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt  --no-cache-dir -r requirements.txt beautifulsoup4


# Copy entire project (so all folders like pages/, media/, etc. come over)
COPY . .
COPY head.html patch_streamlit_head.py /app/

# Streamlit launch command
ENTRYPOINT ["bash", "-lc", "python3 /app/patch_streamlit_head.py && streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0"]