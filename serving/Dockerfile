# The model service in a container; the model stays in Ollama on the host (it needs the GPU).
#   docker build -f serving/Dockerfile -t home-llm-serving .   (from the repo root)
FROM python:3.11-slim
WORKDIR /srv
COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt
COPY llm llm
COPY serving serving
ENV MODEL_SERVICE_HOST=0.0.0.0 MODEL_SERVICE_PORT=8090 MODEL_SERVICE_OLLAMA_URL=http://host.docker.internal:11434 PYTHONUNBUFFERED=1
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')"
CMD ["python", "-m", "serving"]
