# ATLAS in a container (CPU). Train, analyse or serve the web interface:
#   docker build -t atlas .
#   docker run --rm atlas demo
#   docker run --rm -p 5000:5000 -v "$PWD/models:/work/models" atlas serve --host 0.0.0.0
FROM python:3.12-slim
WORKDIR /opt/atlas
COPY pyproject.toml README.md LICENSE ./
COPY atlas ./atlas
RUN pip install --no-cache-dir . && useradd --create-home atlas
USER atlas
WORKDIR /work
ENTRYPOINT ["atlas"]
CMD ["--help"]
