from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import pandas as pd
import seaborn as sns
import joblib
import os


app = FastAPI()

# Configuración de la base de datos
database_username = 'root'
database_password = '98Microne$'
database_host = 'db'
database_port = '3306'
database_name = 'penguin_db'

# Cargar los datos de penguin dataset
df = sns.load_dataset("penguins")

# Eliminar filas con valores faltantes
df.dropna(inplace=True)

# Conexión a la base de datos
connection_string = f"mysql+pymysql://{database_username}:{database_password}@{database_host}:{database_port}/{database_name}"
engine = create_engine(connection_string)

# Crear la base de datos si no existe
if not database_exists(engine.url):
    create_database(engine.url)

engine = create_engine(connection_string)

# Definición de la sesión
SessionLocal = sessionmaker(bind=engine)
# Definición de la clase base para las clases de la base de datos
Base = declarative_base()


# Definición de la clase para la tabla de penguins
class Penguin(Base):
    __tablename__ = "penguins"
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String(16), index=True)
    island = Column(String(16), index=True)
    bill_length_mm = Column(Integer, index=True)
    bill_depth_mm = Column(Integer, index=True)
    flipper_length_mm = Column(Integer, index=True)
    body_mass_g = Column(Integer, index=True)

# Creación de la tabla de penguins en la base de datos
Base.metadata.create_all(bind=engine)

# Definición del modelo para crear nuevos penguins
class PenguinCreate(BaseModel):
    species: str
    island: str
    bill_length_mm: int
    bill_depth_mm: int
    flipper_length_mm: int
    body_mass_g: int

# Endpoint para crear un nuevo penguin en la base de datos
@app.post("/penguins/")
def create_penguin(penguin: PenguinCreate):
    db = SessionLocal()
    db_penguin = Penguin(
        species=penguin.species,
        island=penguin.island,
        bill_length_mm=penguin.bill_length_mm,
        bill_depth_mm=penguin.bill_depth_mm,
        flipper_length_mm=penguin.flipper_length_mm,
        body_mass_g=penguin.body_mass_g
    )
    db.add(db_penguin)
    db.commit()
    db.refresh(db_penguin)
    return db_penguin

# Endpoint para eliminar todos los penguins de la base de datos
@app.delete("/penguins/")
def delete_penguins():
    db = SessionLocal()
    db.query(Penguin).delete()
    db.commit()
    return {"message": "All penguins have been deleted."}

# Endpoint para entrenar un modelo de clasificación y guardar los pesos
@app.post("/train/")
def train_model():
    db = SessionLocal()

    # Obtener los datos de la base de datos
    query = db.query(
        Penguin.species,
        Penguin.island,
        Penguin.bill_length_mm, 
        Penguin.bill_depth_mm, 
        Penguin.flipper_length_mm,
        Penguin.body_mass_g
        )
    df = pd.read_sql(query.statement, db.bind)

        # Target variable can also be encoded using sklearn.preprocessing.LabelEncoder
    df['species']= df['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})

    # Eliminar filas con valores faltantes
    imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 
    df.iloc[:,:] = imputer.fit_transform(df)

    df = df.drop(['island'], axis=1)

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X = df.drop(['species'], axis=1)
    y = df['species']

    model = RandomForestClassifier()
    model.fit(X, y)

    # Guardar los pesos del modelo en la carpeta 'model_weights'
    if not os.path.exists('weights'):
        os.makedirs('weights')
    joblib.dump(model, 'weights/model.joblib')

    # Calcular la precisión del modelo
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    # Retornar la precisión del modelo como una respuesta JSON
    return JSONResponse(content={"accuracy": acc, "message": "The model has been trained."})
