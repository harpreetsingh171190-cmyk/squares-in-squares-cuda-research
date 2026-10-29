FROM python:3.11-slim
WORKDIR /app
COPY requirements-v4.txt requirements-ml.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-v4.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
