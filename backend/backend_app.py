from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Literal
import os
import uuid
from datetime import datetime


from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'), override=True)
# Initialize FastAPI app
app = FastAPI(title="RAG API", description="RAG with Azure AI Search")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to only allow your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model: Optional[Literal["gpt-4o", "gpt4o-mini"]] = "gpt-4o"

# Chat history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#  model names  and deploy names
MODEL_DEPLOYMENTS = {
    "gpt-4o": "gpt-4o",  # Replace with actual deployment name
    "gpt4o-mini": "gpt4o-mini"  # Replace with actual deployment name
}

#  Azure OpenAI chat model
def get_llm(model_name: str):
    """Get Azure OpenAI LLM with specified model"""
    deployment_name = MODEL_DEPLOYMENTS.get(model_name, "gpt-4o")
    return AzureChatOpenAI(
        deployment_name=deployment_name,
        model_name=model_name,
        temperature=0,
        streaming=False,
        api_version="2024-09-01-preview"
    )

#  embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002", 
    model="text-embedding-ada-002", 
    chunk_size=10
    
)

# Connect to Azure AI Search endpoint with key we can change this to erdc endpoint and use managed identity
index_name = "ky-rate-case-index2"  # Replace with erdc index name
vector_store = AzureSearch(
    azure_search_endpoint="https://your-ai-endpoint.search.windows.net",
    azure_search_key="your-key",
    index_name=index_name,
    embedding_function=embeddings.embed_query
)

# Create a retriever with azureaisearchretreiver lib 
retriever = AzureAISearchRetriever(
    content_key="content", top_k=6, index_name="ky-rate-case-index2", service_name= "ppl-test-aisearch1273293761837", api_key=os.environ.get("SEARCH_API_KEY") 
)

# Cache for model-specific RAG chains
rag_chains = {}

def get_rag_chain(model_name: str):
    """Get or create a RAG chain for the specified model"""
    if model_name in rag_chains:
        return rag_chains[model_name]
    
    # Initialize model
    llm = get_llm(model_name)
    
    #  context-aware question reformulation prompt
    contextualize_q_system_prompt = (
        "You are KyRateExpert, specialized in Kentucky utility rate case documentation. "
        "Given the chat history and the latest user question which might reference context in the chat history, "
        "decompose the question and formulate a standalone, comprehensive question that:"
        "\n\n"
        "1. Captures all entities, concepts, and relationships in the original query"
        "2. Expands acronyms where appropriate (e.g., KU = Kentucky Utilities)"
        "3. Includes synonyms or related terms that might enhance retrieval"
        "4. Can be understood without the chat history"
        "\n\n"
        "Do NOT answer the question, just reformulate it to maximize the chance of retrieving relevant rate case information."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    #  QA prompt template
    system_prompt = (
        "You are KyRateExpert, an AI assistant specialized in Kentucky utility rate case documentation for the Kentucky Public Service Commission (PSC). "
        "\n\n"
        "When responding to the user's question, follow these guidelines:"
        "\n\n"
        "1. ANALYZE thoroughly: Carefully examine the retrieved context below to extract relevant information"
        "2. CONNECT THE DOTS: Synthesize information across multiple documents, even when not explicitly asked"
        "3. PROVIDE CONTEXT: Explain regulatory principles and cite specific case numbers when available"
        "4. BE THOROUGH: If you can't find all information requested, provide partial answers rather than saying you don't know"
        "5. SUGGEST RELATED INFO: If the retrieval is incomplete, suggest related information that might be helpful"
        "6. MAINTAIN EXPERTISE: Use a professional, authoritative tone while explaining complex concepts clearly"
        "\n\n"
        "IMPORTANT: For each piece of information you use from the context, cite your source using the format [docX] "
        "where X is the 1-based index of the document in the context list. "
        "For example: 'Kentucky Utilities filed for a rate increase of 10 in May 2023 [doc1].' "
        "\n\n"
        "If you don't know the answer, explain what specific aspects you couldn't find and suggest related information."
        "\n\n"
        "Retrieved context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Cache and return
    rag_chains[model_name] = rag_chain
    return rag_chain

async def process_query(query: str, session_id: Optional[str] = None, model: str = "gpt-4o"):
    """Process a query using RAG with conversation history"""
    # Validate model selection
    if model not in MODEL_DEPLOYMENTS:
        model = "gpt-4o"  # Default to gpt-4o if invalid model provided
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get the right RAG chain for the model
    rag_chain = get_rag_chain(model)
    
    # Wrap the RAG chain with message history management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # run the query through the chain
    try:
        result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        return {
            "answer": result["answer"],
            "session_id": session_id,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # full error for debugging
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        print(f"Error processing query: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Route that handles multiple input formats from forms, strealit or other clients like react/nextjs
@app.api_route("/chat", methods=["GET", "POST"])
async def chat(request: Request):
    """Unified endpoint for chat interactions"""
    # parse and xtract query, session_id, and model based on request type
    if request.method == "GET":
        # Handle GET request
        query = request.query_params.get("query")
        session_id = request.query_params.get("session_id")
        model = request.query_params.get("model", "gpt-4o")
    else:
        # Handle POST request
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Parse JSON body
            body = await request.json()
            query = body.get("query")
            session_id = body.get("session_id")
            model = body.get("model", "gpt-4o")
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            # Parse form data
            form = await request.form()
            query = form.get("query")
            session_id = form.get("session_id")
            model = form.get("model", "gpt-4o")
        else:
            # Unsupported content type
            return {"error": "Unsupported request format"}
    
    # Validate input
    if not query:
        return {"error": "No query provided"}
        
    # Validate model
    if model not in MODEL_DEPLOYMENTS:
        model = "gpt-4o"  # Default to gpt-4o if invalid model
    
    # Process the query
    result = await process_query(query, session_id, model)
    return result

# Endpoint to list available models
@app.get("/models")
async def list_models():
    """Return available model options"""
    return {
        "models": list(MODEL_DEPLOYMENTS.keys()),
        "default": "gpt-4o"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Endpoint to clear chat history
@app.post("/clear")
async def clear_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in store:
        store[session_id] = ChatMessageHistory()
        return {"status": "success", "message": f"Chat history cleared for session {session_id}"}
    return {"status": "error", "message": f"Session {session_id} not found"}

# Endpoint to get chat history
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get chat history for a session"""
    if session_id in store:
        messages = []
        for message in store[session_id].messages:
            messages.append({
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content
            })
        return {"history": messages}
    return {"status": "error", "message": f"Session {session_id} not found"}
# run using this command: uvicorn backend_app:app --reload
# If this file is run directly, start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)