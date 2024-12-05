# Sử dụng base image Python 3.11-slim
FROM python:3.11-slim

# Đặt thư mục làm việc
WORKDIR /app

# Copy toàn bộ nội dung dự án vào container
COPY . /app

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

# Expose các port cần thiết cho cả hai ứng dụng
EXPOSE 8000 

# Khởi động cả hai ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
