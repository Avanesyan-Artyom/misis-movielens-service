FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY train.py ./train.py
COPY data ./data
RUN python train.py --version 0.9.0 && python train.py --version 1.0.0
EXPOSE 8077
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]
