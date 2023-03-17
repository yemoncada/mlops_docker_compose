from fastapi import FastAPI, HTTPException
from src.pred.image_classifier import *
from pydantic import BaseModel

app = FastAPI(title="Image Classifier API")

class Img(BaseModel):
    img_url: str

@app.post("/predict/tf/", status_code=200)
async def predict_tf(request: Img):
    prediction = tf_run_classifier(request.img_url)
    if not prediction:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(
            status_code=404, detail="Image could not be downloaded"
        )
    return prediction

@app.get
def load_database():
    pass

@app.post
def delete_database():
    pass

@app.post("/predict/tf/", status_code=200)
def train_model():
    pass