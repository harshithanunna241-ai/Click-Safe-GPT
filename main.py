
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.services.prediction_service import prediction_service


app = FastAPI(
    title="ClickSafeGPT API",
    description="AI-powered SMS, Email and URL threat detection API",
    version="1.0.0",
)


# Frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SMSRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100000)


class EmailRequest(BaseModel):
    email_text: str = Field(min_length=1, max_length=100000)


class URLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


@app.get("/")
def root():
    return {
        "application": "ClickSafeGPT",
        "status": "running",
        "message": "Cybersecurity threat detection API is ready.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "ClickSafeGPT",
        "models": {
            "sms": prediction_service.sms_model is not None,
            "email": prediction_service.email_model is not None,
            "url": prediction_service.url_model is not None,
        },
    }


@app.post("/predict/sms")
def predict_sms(request: SMSRequest):
    try:
        return prediction_service.predict_sms(request.message)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/predict/email")
def predict_email(request: EmailRequest):
    try:
        return prediction_service.predict_email(request.email_text)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/predict/url")
def predict_url(request: URLRequest):
    try:
        return prediction_service.predict_url(request.url)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
