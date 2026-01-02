#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
from datetime import datetime, timedelta
from aromakiss_rag_bot import AromaKissRAG

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    description="API for content generation using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_bot: Optional[AromaKissRAG] = None

conversations: Dict[str, List[Dict]] = {}
conversation_timestamps: Dict[str, datetime] = {}
CONVERSATION_TIMEOUT = timedelta(hours=2)

class ChatRequest(BaseModel):
    message: str
    message_type: Optional[str] = "general"
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    message_type: str
    status: str = "success"

def cleanup_old_conversations():
    current_time = datetime.now()
    expired_chats = [
        chat_id for chat_id, timestamp in conversation_timestamps.items()
        if current_time - timestamp > CONVERSATION_TIMEOUT
    ]
    for chat_id in expired_chats:
        conversations.pop(chat_id, None)
        conversation_timestamps.pop(chat_id, None)

def get_conversation_context(chat_id: str, max_messages: int = 6) -> str:
    if not chat_id or chat_id not in conversations:
        return ""
    
    recent_messages = conversations[chat_id][-max_messages:]
    context = "\n\n--- КОНТЕКСТ РАЗГОВОРА ---\n"
    for msg in recent_messages:
        role = "Пользователь" if msg["role"] == "user" else "Ты"
        context += f"{role}: {msg['content']}\n"
    
    return context

def store_conversation_message(chat_id: str, role: str, content: str):
    if not chat_id:
        return
    
    if chat_id not in conversations:
        conversations[chat_id] = []
    
    conversations[chat_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
        if len(conversations[chat_id]) > 20:
        conversations[chat_id] = conversations[chat_id][-20:]
    
    conversation_timestamps[chat_id] = datetime.now()

@app.on_event("startup")
async def startup_event():
    global rag_bot
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or as an environment variable."
            )
        
        logger.info("Initializing RAG bot...")
        rag_bot = AromaKissRAG(api_key)
        logger.info("RAG bot initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG bot: {e}")
        raise

@app.get("/")
async def root():
    cleanup_old_conversations()
    return {
        "message": "RAG API is running",
        "status": "healthy",
        "bot_loaded": rag_bot is not None,
        "active_conversations": len(conversations)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG bot not initialized")
    
    try:
        message = request.message.strip()
        message_type = request.message_type
        chat_id = request.chat_id
        
        cleanup_old_conversations()
        
        logger.info(f"Processing message: {message[:50]}... (type: {message_type}, chat: {chat_id})")
        
        if chat_id:
            store_conversation_message(chat_id, "user", message)
        
        context = get_conversation_context(chat_id) if chat_id else ""
        message_lower = message.lower()
        
        if message_type == "post" or message.lower().startswith(('пост:', 'напиши пост', 'создай пост')):
            if message.lower().startswith('пост:'):
                content = message[5:].strip()
            else:
                content = message
            
            response = rag_bot.generate_post(content, conversation_context=context)
            response_type = "post"
            
        elif message_type == "ideas" or message.lower().startswith(('идеи:', 'предложи идеи', 'идеи для постов')):
            if message.lower().startswith('идеи:'):
                theme = message[5:].strip()
            else:
                theme = message.replace('предложи идеи', '').replace('идеи для постов', '').strip()
            
            response = rag_bot.generate_post_ideas(theme, conversation_context=context)
            response_type = "ideas"
            
        elif message_type == "research" or message.lower().startswith(('исследование:', 'расскажи о', 'что такое')):
            if message.lower().startswith('исследование:'):
                topic = message[12:].strip()
            else:
                topic = message
            
            response = rag_bot.research_topic(topic, conversation_context=context)
            response_type = "research"
            
        else:
            if any(keyword in message_lower for keyword in ['пост', 'напиши', 'создай']):
                response = rag_bot.generate_post(message, conversation_context=context)
                response_type = "post"
            elif any(keyword in message_lower for keyword in ['идеи', 'предложи', 'темы']):
                response = rag_bot.generate_post_ideas(message, conversation_context=context)
                response_type = "ideas"
            elif any(keyword in message_lower for keyword in ['расскажи', 'что такое', 'как', 'почему']):
                response = rag_bot.research_topic(message, conversation_context=context)
                response_type = "research"
            else:
                response = rag_bot.conversational_chat(message, context)
                response_type = "conversation"
        
        if chat_id:
            store_conversation_message(chat_id, "assistant", response)
        
        return ChatResponse(
            response=response,
            message_type=response_type,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/generate-post")
async def generate_post(request: ChatRequest):
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG bot not initialized")
    
    try:
        context = get_conversation_context(request.chat_id) if request.chat_id else ""
        response = rag_bot.generate_post(request.message, conversation_context=context)
        return ChatResponse(
            response=response,
            message_type="post",
            status="success"
        )
    except Exception as e:
        logger.error(f"Error generating post: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating post: {str(e)}")

@app.post("/generate-ideas")
async def generate_ideas(request: ChatRequest):
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG bot not initialized")
    
    try:
        context = get_conversation_context(request.chat_id) if request.chat_id else ""
        response = rag_bot.generate_post_ideas(request.message, conversation_context=context)
        return ChatResponse(
            response=response,
            message_type="ideas",
            status="success"
        )
    except Exception as e:
        logger.error(f"Error generating ideas: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating ideas: {str(e)}")

@app.post("/research-topic")
async def research_topic(request: ChatRequest):
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG bot not initialized")
    
    try:
        context = get_conversation_context(request.chat_id) if request.chat_id else ""
        response = rag_bot.research_topic(request.message, conversation_context=context)
        return ChatResponse(
            response=response,
            message_type="research",
            status="success"
        )
    except Exception as e:
        logger.error(f"Error researching topic: {e}")
        raise HTTPException(status_code=500, detail=f"Error researching topic: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_bot_initialized": rag_bot is not None,
        "active_conversations": len(conversations),
        "endpoints": [
            "/chat - Main chat interface",
            "/generate-post - Generate Telegram posts",
            "/generate-ideas - Generate post ideas", 
            "/research-topic - Research topics for content"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RAG API Server...")
    uvicorn.run(
        "rag_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 