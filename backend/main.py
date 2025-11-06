"""
FastAPI server for the multi-agent chatbot system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.workflow import ChatbotWorkflow
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multi-Agent Chatbot API")

# CORS middleware - Configure for production
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    # Add your Netlify URL here after deployment
    os.getenv("FRONTEND_URL", ""),
]

# In production, allow all origins or specify your Netlify domain
if os.getenv("ENVIRONMENT") == "production":
    allowed_origins = ["*"]  # Or specify your exact Netlify domain

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow = ChatbotWorkflow()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    quality_score: float
    timestamp: str


@app.get("/")
async def root():
    return {
        "message": "Multi-Agent Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message through multi-agent workflow
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process through workflow
        result = workflow.process(request.message)
        
        return ChatResponse(
            response=result.get("response", "I'm sorry, I couldn't generate a response."),
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            quality_score=result.get("quality_score", 0.0),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)