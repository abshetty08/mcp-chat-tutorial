FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py live_data_services.py ./
ENV PORT=8002
EXPOSE 8002
CMD ["python", "app.py"]
