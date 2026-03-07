from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from rag_app.retrival import create_index
from rag_app.llm import llm_response

app = FastAPI(title="Medical RAG API")


# Connect to existing Qdrant storage
client = QdrantClient(path="rag_app/qdrant_storage")

# Load existing index
index = create_index(client, "medical_paper")

# Create query engine once
query_engine = index.as_query_engine(similarity_top_k=3)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "RAG API running"}


@app.post("/query")
def query_rag(request: QueryRequest):

    response = query_engine.query(request.query)

    # extract context
    context = "\n\n".join([node.node.text for node in response.source_nodes])

    answer = llm_response(context, request.query)

    return {"query": request.query, "answer": answer, "context": context}
