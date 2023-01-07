import json
from typing import List
import pandas as pd
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from selling_price_prediction import prediction

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.get("/")
async def root(request: Request, message='Samples'):
    return templates.TemplateResponse('index.html',
                                      {"request": request,
                                       "message": message})


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    values = [v for k, v in dict(item).items()]
    columns = [k for k, v in dict(item).items()]
    sample = pd.DataFrame(data=values).T
    sample.columns = columns
    sample_pred = prediction(sample)
    return sample_pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    sample_pred = prediction(items)
    return sample_pred


@app.post("/one_object")
def upload_single(name: str = Form()):
    print(name)
    name = json.loads(name)
    y_pred = predict_item(name)

    return {'price': y_pred}


@app.post("/more_then_one_object")
async def upload(request: Request, name: str = Form(...), item_file: UploadFile = File(...)):
    file_name = '_'.join(name.split()) + '.csv'
    save_path = f'results/{file_name}'
    sample = pd.read_csv(item_file.file)
    sample['prediction_price'] = predict_items(sample)
    sample.to_csv(save_path, index=False)
    return FileResponse(save_path, filename=file_name)
