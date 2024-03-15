from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import torch

from app.hacker_news_deployed_code_v1 import encode_and_pool, MyModel, hn_standard_word2vec_model

# Create an instance of the FastAPI class
app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    input_sentences = [sentence.split() for sentence in input]
    input_embeddings = encode_and_pool(input_sentences, word2vec_model)
    input_features = input_embeddings.clone().detach()
    yield       
    # Example: Load a machine learning model, establish database connections, etc.


app = FastAPI(lifespan=lifespan)

# Define a route for the root URL
@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

@app.post("/upvotes")
async def read_root(request: Request):
    body = await request.body()     
    output = model(input_features)
    
    # feed body through model
    # output prediction
    
    return {"message": "prediction", "body": body}