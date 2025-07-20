FROM tobix/pywine:3.9

RUN wine pip install MetaTrader5 rpyc

COPY mt5_server.py /app/mt5_server.py
WORKDIR /app

CMD ["wine", "python", "mt5_server.py"]