from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import random
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://ec2-54-160-110-158.compute-1.amazonaws.com:5000/")

logged_model_all_g1 = mlflow.pyfunc.load_model("models:/all_crop_group1_model/Production")
logged_model_all_g2 = mlflow.pyfunc.load_model("models:/all_crop_group2_model/Production")
logged_model_all_g3 = mlflow.pyfunc.load_model("models:/all_crop_group3_model/Production")

logged_model_top_g1 = mlflow.pyfunc.load_model("models:/top_crop_group1_model/Production")
logged_model_top_g2 = mlflow.pyfunc.load_model("models:/top_crop_group2_model/Production")
logged_model_top_g3 = mlflow.pyfunc.load_model("models:/top_crop_group3_model/Production")

client = mlflow.MlflowClient()
registered_models = client.search_registered_models()

hw_models = {}
hw_models['all_crops'] = {}
hw_models['top_crop'] = {}
for model in registered_models:
    if model.name.startswith("hw_") and 'all_crop' in model.name:
        name = model.name.replace("hw_", "").replace("_all_crop_ts", "")
        hw_models['all_crops'][name] = mlflow.pyfunc.load_model(f"models:/{model.name}/Production")

    if model.name.startswith("hw_") and 'top_crop' in model.name:
        name = model.name.replace("hw_", "").replace("_top_crop_ts", "")
        hw_models['top_crop'][name] = mlflow.pyfunc.load_model(f"models:/{model.name}/Production")

print('all_crop',hw_models['all_crops'].keys())
print('top_crop',hw_models['top_crop'].keys())

init_year = 2000

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://lark-social-tarpon.ngrok-free.app",
                   ],  # Allow requests from the frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class PredictionInput(BaseModel):
    country: str
    province: str
    year: int
    t2m: Dict[str, float]
    rh2m: Dict[str, float]
    ws10m: Dict[str, float]
    frost_days: Dict[str, float]
    snodp: Dict[str, float]
    ps: Dict[str, float]
    pw: Dict[str, float]
    cropType: str

class InputHW(BaseModel):
    country: str
    crop_type: str
    
class PredictionInputHW(BaseModel):
    country: str
    crop_type: str
    years: List[int]

@app.post("/predict-by-province-manual")
async def predict(input: PredictionInput):
    cropType = input.cropType
    print(cropType)
    # DataFrame with input data
    df_t2m = pd.DataFrame.from_dict(input.t2m, orient='index').T
    df_rh2m = pd.DataFrame.from_dict(input.rh2m, orient='index').T
    df_ws10m = pd.DataFrame.from_dict(input.ws10m, orient='index').T
    df_frost_days = pd.DataFrame.from_dict(input.frost_days, orient='index').T
    df_snodp = pd.DataFrame.from_dict(input.snodp, orient='index').T
    df_ps = pd.DataFrame.from_dict(input.ps, orient='index').T
    df_pw = pd.DataFrame.from_dict(input.pw, orient='index').T
    df = pd.concat([df_t2m, df_rh2m, df_ws10m, df_frost_days, df_snodp, df_ps, df_pw], axis=1)
    
    df.columns = [k.replace('days_','days_M') if 'frost_days' in k else k.replace('_', '_M') for k in df.columns]
    
    df['country'] = input.country.lower()
    df['province'] = input.province.lower()
    df['year'] = input.year
    
    # This is a mock prediction. In a real scenario, you would use a trained model here.
    prediction = random.uniform(0, 100)
    if cropType == 'allCrops':
        if input.country.lower() == 'france':
            prediction = logged_model_all_g1.predict(data=df)[0]
        elif input.country.lower() in ['italy','türkiye','poland','spain']:
            prediction = logged_model_all_g2.predict(data=df)[0]
        else:
            prediction = logged_model_all_g3.predict(data=df)[0]
        
        return {"prediction": round(prediction, 2)}
    
    elif cropType == 'topCrop':
        if input.country.lower() == 'france':
            prediction = logged_model_top_g1.predict(data=df)[0]
        elif input.country.lower() in ['italy','türkiye','poland','spain']:
            prediction = logged_model_top_g2.predict(data=df)[0]
        else:
            prediction = logged_model_top_g3.predict(data=df)[0]
        
        return {"prediction": round(prediction, 2)}
    
    else:
        return {"prediction": -999}

@app.post("/hw-yield-data")
def get_base_data(input: InputHW):
    country = input.country.lower()
    crop_type = input.crop_type
    print(country, crop_type)
    
    try:
        model = hw_models[crop_type][country]
        m_data = model.unwrap_python_model().model.data.endog
        base_data = {k: v for k, v in zip([init_year+i for i in range(len(m_data))], m_data)}
    except Exception as e:
        print(e)
        base_data = {}

    return base_data
  
@app.post("/hw-forecast")
def forecast_data(input: PredictionInputHW):
    country = input.country.lower()
    crop_type = input.crop_type
    years = input.years
    print(country, crop_type, years)
    
    try:
        model = hw_models[crop_type][country]
        m_data = model.unwrap_python_model().model.data.endog
        # max_base = max([init_year+i for i in range(len(m_data))]) + 1
        # pred_years = [i for i in range(max_base, years+1)]
        base_data = {k: v for k, v in zip([min(years)-1] + years, [m_data[-1]] + list(model.predict(years)))}       
    except Exception as e:
        print(e)
        base_data = {}

    return base_data
  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run(app, host="127.0.0.1", port=8000)

