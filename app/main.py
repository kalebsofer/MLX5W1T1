from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import json

from .preprocess import preprocess
from .predict import predict

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


class Article(BaseModel):
    title: str
    author: str
    url: str
    date: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_article(
    request: Request,
    title: str = Form(...),
    author: str = Form(...),
    url: str = Form(...),
    date: str = Form(...),
):
    article = Article(title=title, author=author, url=url, date=date)
    feature_list = preprocess(article.dict(), "title")
    prediction = predict(feature_list)

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "article": article, "prediction": prediction},
    )


@app.post("/upload_json")
async def upload_json_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    json_data = json.loads(contents)

    article = Article(**json_data)
    feature_list = preprocess(article.dict(), "title")
    prediction = predict(feature_list)

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "article": article, "prediction": prediction},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
