FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY . .

EXPOSE 7860

CMD ["python", "server/app.py"]
