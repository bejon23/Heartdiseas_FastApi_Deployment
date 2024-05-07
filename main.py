import os
import pickle
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Load the machine learning model with error handling
model_path = os.path.join(os.path.dirname(__file__), "svm_clf.pkl")
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("Model file 'svm_clf.pkl' not found. Please ensure the model file exists.")

# Set the template folder path
template_folder = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=template_folder)

# Mount the 'static' directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    BMI: float
    Smoking: float
    AlcoholDrinking: float
    Stroke: float
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: float
    Sex: str
    AgeCategory: str
    Race: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str
    SleepTime: float
    Asthma: str
    KidneyDisease: str
    SkinCancer: str

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request, item: Item):
    try:
        # Your prediction logic here
        features = [[
            item.BMI, item.Smoking, item.AlcoholDrinking, item.Stroke,
            item.PhysicalHealth, item.MentalHealth, item.DiffWalking,
            item.Sex, item.AgeCategory, item.Race, item.Diabetic,
            item.PhysicalActivity, item.GenHealth, item.SleepTime,
            item.Asthma, item.KidneyDisease, item.SkinCancer
        ]]  # Adjust as per your model's input requirements
        prediction = model.predict(features)
        return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
