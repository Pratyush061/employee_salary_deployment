from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import pandas as pd


with open("../model.pkl", 'rb') as f:
    model = pickle.load(f)



app = FastAPI()



# Set up templates
templates = Jinja2Templates(directory="templates")

class EmployeeInput(BaseModel):
    age: float
    gender: object
    education_level: object
    job_title: object
    yoe: float


class EmployeePrediction(BaseModel):
    predicted_salary: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=EmployeePrediction)
async def predict(
    request: Request,
    age: float = Form(...),
    gender: object = Form(...),
    education_level: object = Form(...),
    job_title: object = Form(...),
    yoe: float = Form(...),
):
    # Convert the input data to a pandas dataframe
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': yoe
    }])

    # Make a prediction
    predicted_salary = model.predict(input_data)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "predicted_salary": predicted_salary,
            'Age': age,
            'Gender': gender,
            'Education Level': education_level,
            'Job Title': job_title,
            'Years of Experience': yoe
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


