FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["python", "server/app.py"]
