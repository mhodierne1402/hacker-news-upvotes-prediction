from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import random

from app.hacker_news_deployed_code_v1 import encode_and_pool, MyModel, word2vec_model
from app.hacker_news_deployed_code_v1 import *

# Create an instance of the FastAPI class
app = FastAPI()

input_features = None

# This code will run before the application starts.
# Create the model instance
input_size = word2vec_model.vector_size
hidden_size1 = 64
hidden_size2 = 32
model = MyModel(input_size, hidden_size1, hidden_size2)

# Load the saved model state dictionary
model.load_state_dict(torch.load('reg_model.pth'))

# Set the model to evaluation mode
model.eval()

class UpvoteRequest(BaseModel):
    sentence: str

# Define a route for the root URL
@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

@app.post("/upvotes")
async def read_root(request: UpvoteRequest):
    input = request.sentence
    input_embeddings = encode_and_pool(input, word2vec_model)
    input_features = input_embeddings.clone().detach().to(torch.float32)
    
    output = model(input_features)
    
    return {"message": "prediction", "result": random.randint(0, 100)}