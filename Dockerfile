# Dashboard-only image: reads the committed reports/, never calls an API.
FROM python:3.12-slim
WORKDIR /app
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY app app
COPY src src
COPY reports reports
COPY .streamlit .streamlit
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
