# Set up Python Environment
FROM python:3.10

# Install build dependencies
RUN apt update && \
    apt install -y default-jdk && \
    apt install -y gcc && \
    apt install -y zip && \
    apt install -y nano

# We install the Java JDK for opening microscope images, gcc for compilation of the python-javabridge package, zip for packaging the output, and nano for editing files in the container.

# Set JAVA_HOME and update PATH
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"

# Then install your package
WORKDIR /app
RUN mkdir /app/FlickerPrint

# Install some of the more time-consuming dependencies first
# These have to be installed before the COPY line invalidates the cache on changed files
RUN python3 -m pip install \
    "tensorboard==2.19.0" "tensorboard-data-server==0.7.2" "tensorflow==2.19.0" \
    "tensorflow-io-gcs-filesystem==0.37.1" "tensorflow-probability==0.25.0" "numexpr==2.11.0" \
    "matplotlib==3.10.3" "numpy==1.26.4" "h5py==3.14.0" \
    "trieste==0.13.2" "gpflow==2.10.0" "gpflux==0.2.3"

COPY . /app/FlickerPrint

RUN cd /app/FlickerPrint/src && \
    python3 -m pip install . && \
    cd /app

CMD ["bash"]
