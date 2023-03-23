FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

# Copiar los archivos necesarios
COPY load_data.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8000

# Ejecutar la aplicación
CMD ["uvicorn", "load_data:app", "--host", "0.0.0.0", "--port", "8000"]