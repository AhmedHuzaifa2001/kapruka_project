from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import agents
from fastapi.responses import StreamingResponse

# 1. Initialize the web server
app = FastAPI(title="Kapruka AI Backend")

# 2. Allow frontend applications to talk to this server (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define what the incoming data from the user should look like
class ChatRequest(BaseModel):
    message: str

# 4. Create the web endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n[API] Received message: {request.message}")
    
    # Send the final string back to the user as JSON
    return StreamingResponse(
        agents.chat_stream(request.message), 
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    try:
        import uvicorn

        uvicorn.run("main:app", host="localhost", port=8000, reload=True)
    except Exception as e:
        print("Failed to start server with uvicorn. Is uvicorn installed?", e)