from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
import os

app = FastAPI()

# Cargar el modelo entrenado en la API
# model = load(os.path.join(
#     os.path.expanduser('~'),
#     "mlops_docker_compose/weights/model.joblib"))

model = load('weights/model.joblib')

# Definir el esquema de entrada de datos
class PenguinData(BaseModel):
    bill_length_mm: int
    bill_depth_mm: int
    flipper_length_mm: int
    body_mass_g: int

# Crear una ruta para la inferencia
@app.post("/predict/")
def predict_species(penguin: PenguinData):
    # Procesar los datos de entrada
    X = [[penguin.bill_length_mm,
          penguin.bill_depth_mm,
          penguin.flipper_length_mm,
          penguin.body_mass_g]]
    
    # Realizar la inferencia
    species = model.predict(X)
    
    # Devolver la predicción de la especie del pingüino
    if species[0] == 0:
        return {"species": "Adelie"}
    elif species[0] == 1:
        return {"species": "Chinstrap"}
    elif species[0] == 2:
        return {"species": "Gentoo"}
    else:
        # En caso de que se produzca un error, lanzar una excepción HTTP
        raise HTTPException(status_code=400, detail="Invalid species prediction.")