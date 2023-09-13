import json
import sys
from pathlib import Path
import yaml
import uvicorn
from pydantic import BaseModel

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from typing import List

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

app = FastAPI()
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(src_path.joinpath('params.yaml')) as conf_file:
        config = yaml.safe_load(conf_file)

model_path = config["train"]["model_path"]
model = load(filename=src_path.joinpath(model_path))

class Request(BaseModel):
     data: List

@app.post("/predict")
async def predict(request_data: Request = Body(..., example={})):
    input_data = json.loads(request_data.model_dump_json())
    predicted = model.predict(input_data["data"])
    return predicted.tolist()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
