from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from src.logger import logger, log_event
from src.document_processor import DocumentProcessor
from src.retriever import HybridRetriever
from src.agent import ConversationalAgent

# Global state
processor = DocumentProcessor()
retriever = HybridRetriever()
agent: Optional[ConversationalAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_event("System Initialization", {"Status": "Starting up ZX Bank AI Backend..."})
    
    # Process markdown documents and build hybrid index on startup
    chunks = processor.process_documents()
    retriever.build_index(chunks)
    
    # Initialize the Agent
    global agent
    agent = ConversationalAgent(retriever)
    
    log_event("System Ready", {"Status": "Waiting for AI chat requests."})
    yield

app = FastAPI(title="ZX Bank Conversational AI Backend", lifespan=lifespan)

# Mount the static directory to serve HTML/CSS/JS
import os
os.makedirs("static", exist_ok=True)
# Add CORS middleware to prevent browser blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if agent is None:
            logger.error("Agent not initialized!")
            raise HTTPException(status_code=503, detail="AI Agent is still initializing. Please wait 5 seconds and refresh.")
            
        log_event("Incoming Request", {"Session": request.session_id, "Query": request.query})
        
        # Process the request through our routing and RAG agent
        response_text = agent.process_request(request.session_id, request.query)
        
        log_event("Final Output", {"Response Sample": response_text[:100] + ("..." if len(response_text) > 100 else "")})
        return ChatResponse(response=response_text)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"CRITICAL API ERROR: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
