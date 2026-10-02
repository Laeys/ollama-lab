FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY client.py .
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://ollama:11434
ENV MODEL=smollm2:135m-instruct-q2_K
USER 10001:10001
CMD ["python", "client.py"]