# Usa Python 3.12.3 come immagine di base
FROM python:3.12.3

ENV WD = /app

# Imposta la directory di lavoro
WORKDIR ${WD}

# Aggiorna il sistema e installa librerie necessarie
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia il file requirements.txt nella cartella di lavoro
COPY requirements.txt  /requirements.txt

# Installa le librerie richieste
#RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia il resto del codice sorgente nella cartella di lavoro
#COPY . 
