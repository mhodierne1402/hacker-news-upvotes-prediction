from fastapi import FastAPI, Request

# Create an instance of the FastAPI class
app = FastAPI()

# Define a route for the root URL
@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

@app.post("/upvotes")
async def read_root(request: Request):
    body = await request.body()        
    
    # feed body through model
    # output prediction
    
    return {"message": "prediction", "body": body}