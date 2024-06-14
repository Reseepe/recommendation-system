# app/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import find_similar_recipes

class UserInput(BaseModel):
    user_input: str

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Recipe Recommendation API'}

@app.post('/recommend')
def recommend(data: UserInput):
    user_input = data.user_input
    recommendations = find_similar_recipes(user_input)
    return {'recommended_recipes': recommendations}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)