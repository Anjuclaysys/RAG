Task: Implement a RAG Pipeline Using LangChain + HuggingFace + Qdrant + LlamaIndex
(Implementation to be done using Jupyter Notebook – .ipynb)

--------------------------------------------------

Objective:

Implement a Retrieval Augmented Generation (RAG) pipeline that demonstrates
the full lifecycle of document ingestion, embedding storage, retrieval,
and LLM-based answer generation.

The implementation must use different tools for different stages of the pipeline
to understand how each framework operates.

--------------------------------------------------

Tech Stack (Mandatory)

- Jupyter Notebook (.ipynb)
- LangChain (for ingestion and text splitting)
- HuggingFace Transformers (for embeddings)
- Qdrant (Vector Database)
- LlamaIndex (for retrieval)
- Google Gemini API (for answer generation)
- Python

--------------------------------------------------

API Requirement

Use Google Gemini Free API.

Steps:

1. Create a personal Google account if not already available.
2. Generate a free API key from Google AI Studio:
   https://aistudio.google.com/
3. Store the API key as an environment variable.
4. Do NOT hardcode API keys inside the notebook.

Example:

Set environment variable:

export GEMINI_API_KEY="your_api_key"

--------------------------------------------------

Phase 1 – Document Ingestion

Use LangChain to load documents.

Requirements:

- Load at least one document source
  (PDF / TXT / Markdown)
- Extract text from document
- Print raw extracted content
- Ensure document metadata is preserved

Expected Output:

- Successfully loaded document
- Raw document text preview

--------------------------------------------------

Phase 2 – Text Chunking

Use LangChain Text Splitters.

Tasks:

- Implement RecursiveCharacterTextSplitter
- Experiment with different chunk sizes
- Add chunk overlap
- Print resulting chunks

Questions to Answer:

- Why is chunk overlap needed?
- What happens if chunk size is too large?

Expected Output:

- List of text chunks
- Demonstration of chunk boundaries

--------------------------------------------------

Phase 3 – Embedding Generation

Use HuggingFace Transformer embeddings.

Tasks:

- Use a HuggingFace embedding model
  (example: sentence-transformers/all-MiniLM-L6-v2)
- Generate embeddings for each chunk
- Inspect embedding vector size
- Print embedding shape

Expected Output:

- Embedding vectors generated for each chunk
- Embedding dimension explanation

--------------------------------------------------

Phase 4 – Vector Storage (Qdrant)

Store embeddings inside Qdrant.

Tasks:

- Setup Qdrant locally
- Create a collection
- Store chunk embeddings
- Store metadata (source, chunk index)

Verify:

- Number of stored vectors
- Collection configuration

Expected Output:

- Embeddings successfully stored in Qdrant
- Retrieval-ready vector database

--------------------------------------------------

Phase 5 – Retrieval Using LlamaIndex

Use LlamaIndex to query stored vectors.

Tasks:

- Connect LlamaIndex to Qdrant collection
- Create vector index
- Implement query engine
- Retrieve top-k relevant chunks

Test Queries:

- Ask at least 3 questions based on document
- Display retrieved chunks
- Display similarity scores

Expected Output:

- Correct chunks retrieved
- Demonstrate semantic retrieval

--------------------------------------------------

Phase 6 – Response Generation Using Gemini

Use Google Gemini API to generate answers.

Tasks:

- Retrieve top-k chunks
- Combine chunks into context
- Create prompt with retrieved context
- Send prompt to Gemini model
- Generate answer

Requirements:

- Print retrieved chunks
- Print full prompt sent to Gemini
- Print final answer

--------------------------------------------------

Phase 7 – Observability

Add debugging and visibility into pipeline.

Display:

- Chunk sizes
- Number of embeddings generated
- Stored vector count
- Top-k retrieved chunks
- Similarity scores
- Final prompt sent to Gemini

--------------------------------------------------

Notebook Structure

1. Introduction
2. Document Loading
3. Text Chunking
4. Embedding Generation
5. Qdrant Vector Storage
6. Retrieval using LlamaIndex
7. Gemini Answer Generation
8. Observations / Learnings

--------------------------------------------------