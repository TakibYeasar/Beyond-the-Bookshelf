import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.recommender import get_recommendations_logic, RecommendationRequest
from api.data_loader import categories, tones

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """
    Serves the main HTML dashboard page with dropdown options.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": categories,
            "tones": tones
        }
    )


@app.post("/recommendations")
def recommend_books_api(request_body: RecommendationRequest):
    """
    API endpoint to get book recommendations based on user input.
    """
    return get_recommendations_logic(
        request_body.query,
        request_body.category,
        request_body.tone
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
