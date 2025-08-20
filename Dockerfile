# Usa Ubuntu 22.04 como imagen base, una versión moderna y estable.
FROM ubuntu:22.04

# Configura el frontend de Debian para evitar la interacción del usuario durante las instalaciones.
ENV DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# PASO 1: Instalar dependencias del sistema y Python
# ==============================================================================
# Se actualizan los repositorios y se instalan herramientas esenciales, Python 3.10,
# y las librerías de desarrollo necesarias para SUMO y otras herramientas.
# SOLUCIÓN CLAVE: Se instala 'libsqlite3-dev' junto a 'libgdal-dev' para prevenir
# el conflicto de símbolos que causa el warning de libsumo.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    wget git curl unzip build-essential \
    libsqlite3-0 libsqlite3-dev \
    libgdal-dev gdal-bin \
    # Limpia la caché de apt para reducir el tamaño de la imagen.
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configura 'python3.10' y 'pip3' como los comandos por defecto para 'python' y 'pip'.
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# ==============================================================================
# PASO 2: Instalar SUMO
# ==============================================================================
# Se añade el repositorio personal (PPA) oficial de SUMO para obtener una versión estable y reciente.
# Esto asegura que las versiones de sumolib y traci sean >= 1.14.0.
RUN add-apt-repository ppa:sumo/stable -y && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ==============================================================================
# PASO 3: Configurar variables de entorno para SUMO
# ==============================================================================
# Estas variables son cruciales para que Python y las librerías de RL encuentren
# las herramientas de SUMO (traci, sumolib).
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH=/usr/share/sumo/tools
ENV LIBSUMO_AS_TRACI=1

# ==============================================================================
# PASO 4: Instalar librerías de Python con versiones específicas
# ==============================================================================
# Se actualiza pip y luego se instalan todas las librerías de Python requeridas.
# Se usa '--no-cache-dir' como buena práctica para mantener la imagen ligera.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    # --- Versiones exactas requeridas ---
    ray[rllib,tune]==2.7.0 \
    numpy==1.23.4 \
    # --- Versiones mínimas requeridas ---
    Pillow>9.4.0 \
    supersuit>3.9.0 \
    torch>1.13.1 \
    tensorflow-probability>0.19.0 \
    gymnasium>=0.28 \
    pettingzoo>=1.24.3 \
    # --- Librerías sin versión específica ---
    sumo-rl \
    pandas \
    matplotlib \
    dm-tree \
    scipy \
    lz4 \
    stable-baselines3>=2.0.0 \  
    pyvirtualdisplay\
	pytest



# ==============================================================================
# PASO 5: Configurar el entorno de trabajo
# ==============================================================================
# Se crea y establece el directorio de trabajo dentro del contenedor.
WORKDIR /workspace

# Comando por defecto que se ejecuta al iniciar el contenedor.
# Informa al usuario que el entorno está listo.
#CMD ["echo", "Contenedor listo. Monta tu código en /workspace y ejecuta tu script de Python."]
CMD ["/bin/bash"]
