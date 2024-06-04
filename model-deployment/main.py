from fastapi import FastAPI
import joblib
from pydantic import BaseModel



model = joblib.load('models/knn_model.joblib')
scaler = joblib.load('models/scaler.joblib')


app = FastAPI()

@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"

items = {}


# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}


# get request
@app.get("/items")
def create_item(item: str):
    print(item)
    items[item] = 0
    return items


@app.get("/update")
def update_item(item: str, value: int):
    if items.get(item, None) == None:
        return "error: item not found"

    items[item] = value
    return items[item]


@app.get("/all")
def get_items_item():
    return items



# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    age: float
    appearance: int
    minutes_played: int
    days_injured: int
    games_injured: int
    award: int
    current_value: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
    'age': input_features.age,
    'appearance': input_features.appearance,
    'minutes_played': input_features.minutes_played,
    'days_injured': input_features.days_injured,
    'games_injured': input_features.games_injured,
    'award': input_features.award,
    'current_value': input_features.current_value,

    }
    
    
    # Convert dictionary values to a list in the correct order
    # features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler.transform([list(dict_f.values())])
    
    return scaled_features

def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    
@app.post("/predict")

async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}
