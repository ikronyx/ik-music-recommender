from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from model.recommender import recommend_by_mood

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("data/SpotifyFeatures.csv")

@app.get("/recommend")
def recommend(mood: str = Query(..., description="User mood e.g. happy, sad")):
    results = recommend_by_mood(df, mood)
    return results