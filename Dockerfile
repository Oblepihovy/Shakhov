FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем проект
COPY . .

# чтобы src импортировался
ENV PYTHONPATH=/app

# дефолтный запуск (переопределишь в CLI)
CMD ["bash"]