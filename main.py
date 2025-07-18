from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model - simple path without os module
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")

class EmployeeInput(BaseModel):
    age: float
    gender: str
    education_level: str
    job_title: str
    yoe: float

class EmployeePrediction(BaseModel):
    predicted_salary: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(...),
    gender: str = Form(...),
    education_level: str = Form(...),
    job_title: str = Form(...),
    yoe: float = Form(...),
):
    # Convert the input data to a pandas dataframe
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education_Level': education_level,
        'Job_Title': job_title,
        'Years_of_Experience': yoe
    }])

    # Make a prediction
    predicted_salary = model.predict(input_data)[0]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "predicted_salary": round(predicted_salary, 2),
            'Age': age,
            'Gender': gender,
            'Education_Level': education_level,
            'Job_Title': job_title,
            'Years_of_Experience': yoe
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
