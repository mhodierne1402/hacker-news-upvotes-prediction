from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sentencepiece as spm

from app.hacker_news_deployed_code_v1 import encode_and_pool, word2vec_model
from app.hacker_news_deployed_code_v1 import *

# Hyperparameters
hidden_dim_1= 64
hidden_dim_2 = 32
num_epochs = 100
vocab_size = 4000
embedding_dim = 300
learning_rate = 0.001
EMBED_MAX_NORM = 1

# Define the SkipGramModel class
class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_norm=1):
        super(SkipGram_Model, self).__init__()
        self.vocab_size = 4000
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, max_norm=max_norm)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
    
    def embed(self, inputs_):
        # This method will return just the embeddings
        return self.embeddings(inputs_)

class ScorePredictor(nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, embedding_model):
        super(ScorePredictor, self).__init__()
        self.embedding_model = embedding_model
        
        # Use embedding_model's embedding dimension
        embedding_dim = self.embedding_model.embedding_dim
        
        self.hidden_1 = nn.Linear(embedding_dim, hidden_dim_1)
        self.hidden_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.output = nn.Linear(hidden_dim_2, 1)
        self.relu = nn.ReLU()

    def forward(self, titles):
        embeddings = self.embedding_model.embed(titles)
        pooled_embeddings = embeddings.mean(dim=1)
        hidden_1_output = self.relu(self.hidden_1(pooled_embeddings))
        hidden_2_output = self.relu(self.hidden_2(hidden_1_output))
        score_predictions = self.output(hidden_2_output)
        return score_predictions

# Create an instance of the FastAPI class
app = FastAPI()

pretrained_model_path = './model.pt'
embedding_model = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
embedding_model.eval()  # Assuming you're doing inference

# Manually set the embedding_dim if it's not part of the loaded state_dict
embedding_model.embedding_dim = 300

# Manually add the embed method to the loaded model if it's missing
def embed(self, inputs_):
    return self.embeddings(inputs_)

# Assign the method to the instance
embedding_model.embed = embed.__get__(embedding_model, SkipGram_Model)

# Freeze the parameters (weights) of the embedding model
for param in embedding_model.parameters():
    param.requires_grad = False

# Define the path to the combined model
combined_model_path = './combined_model.pth'

# Load the combined state dictionary
combined_state = torch.load(combined_model_path, map_location=torch.device('cpu'))

score_predictor_model = ScorePredictor(hidden_dim_1, hidden_dim_2, embedding_model)
score_predictor_model.load_state_dict(combined_state['score_predictor_state_dict'])
score_predictor_model.eval()

def score_predictor(title, score_predictor_model):
    # Tokenize the title using the sentencepiece tokenizer
    sp = spm.SentencePieceProcessor(model_file='./techcrunch_sp.model')
    title_tokens = sp.encode_as_pieces(title)
    title_ids = [sp.piece_to_id(token) for token in title_tokens]

    # Convert to tensor and add batch dimension
    title_tensor = torch.tensor(title_ids).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():  # Ensure no gradients are computed
        score = score_predictor_model(title_tensor)

    return score.item()

class UpvoteRequest(BaseModel):
    sentence: str

# Define a route for the root URL
@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

# @app.post("/upvotes")
# async def read_root(request: UpvoteRequest):
#     input = request.sentence
#     input_embeddings = encode_and_pool(input, word2vec_model)
#     input_features = input_embeddings.clone().detach().to(torch.float32)
    
#     output = model(input_features)
    
#     return {"message": "prediction", "result": random.randint(0, 100)}

@app.post("/upvotes-maria")
async def read_root(request: UpvoteRequest):
    predicted_score = score_predictor(request.sentence, score_predictor_model)
    print("Predicted score:", predicted_score)
    
    return {"message": "prediction", "result": predicted_score}