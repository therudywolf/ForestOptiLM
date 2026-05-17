# SPDX-License-Identifier: AGPL-3.0-or-later
FROM python:3.12-slim

LABEL maintainer="therudywolf <https://github.com/therudywolf>"
LABEL description="ForestOptiLM / Nocturne Data Forge"
LABEL license="AGPL-3.0-or-later"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 \
        libfontconfig1 libfreetype6 tk && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# GUI (default)
CMD ["python", "main.py"]
# Headless example:
# docker run --rm -v %cd%:/data forestoptilm python -m forestoptilm.cli analyze /data -q "..." -o /data/out.md
