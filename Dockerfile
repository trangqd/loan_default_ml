FROM apache/airflow:2.9.2-python3.12

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1️⃣  Install Java 17 (for PySpark) and procps (handy for `ps`)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

# 2️⃣  Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 3️⃣  Project code
COPY utils /opt/airflow/utils
COPY dags  /opt/airflow/dags
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"


