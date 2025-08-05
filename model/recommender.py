from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from .mood_to_valence_map import mood_valence_dict

def recommend_by_mood(df: pd.DataFrame, mood: str, n=5):
    if mood.lower() not in mood_valence_dict:
        return {"error": "Unsupported mood"}

    target_valence = mood_valence_dict[mood.lower()]
    features = df[["valence", "energy", "danceability"]]
    similarity = cosine_similarity([[target_valence, 0.5, 0.5]], features)
    top_indices = similarity[0].argsort()[-n:][::-1]

    return df.iloc[top_indices][[
        "track_name", "artist_name", "genre", "valence", "energy", "danceability"
    ]].to_dict(orient="records")