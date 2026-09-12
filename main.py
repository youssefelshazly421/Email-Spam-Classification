from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

feature_extraction = pickle.load(open("feature_extraction.pkl", "rb"))
model = pickle.load(open("logistic_regression.pkl", "rb"))

@app.post("/predict")
def predict(email: str):
    email_vector = feature_extraction.transform([email])
    prediction = model.predict(email_vector)

    if prediction[0] == 0:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return {"result": result}