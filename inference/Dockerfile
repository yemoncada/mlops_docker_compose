FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

# Copiar los archivos necesarios
COPY class_inference.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8001

# Ejecutar la aplicación
CMD ["uvicorn", "class_inference:app", "--host", "0.0.0.0", "--port", "8001"]