from fastapi import FastAPI
from inference_model.infer import predict

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Bank Inference API Running"}

@app.post("/run-inference")
def run():
    result = predict("bank-data/inference_data.csv")
    result.to_csv("bank-data/output.csv", index=False)
    return {"status": "Inference completed"}
