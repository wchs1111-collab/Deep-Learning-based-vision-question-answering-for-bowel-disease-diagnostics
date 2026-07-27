from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ite/{item_i}/{q}")
def read_ite(item_i: int, q: str):
    return {"item_id": item_i, "p": q}