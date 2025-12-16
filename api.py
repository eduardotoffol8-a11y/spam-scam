from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Inicializa a aplicação FastAPI
app = FastAPI(title="Detector de Spam API")

# Habilita CORS para permitir chamadas do navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restrinja ao domínio do site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada para JSON {"message": "..."}
class MessageIn(BaseModel):
    message: str

# Carrega modelo e vetor previamente salvos
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.post("/predict")
def predict(payload: MessageIn):
    """
    Recebe JSON {"message": "..."} e retorna se é spam ou não.
    """
    message = payload.message
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    return {
        "message": message,
        "prediction": "spam" if pred == 1 else "ham"

    }
