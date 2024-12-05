FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

EXPOSE 8000 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
