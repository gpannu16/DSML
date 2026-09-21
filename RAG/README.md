# RAG with Hugging Face Documentation and Milvus

This project implements a Retrieval-Augmented Generation (RAG) pipeline using the Hugging Face documentation dataset, Milvus vector search, sentence embeddings, and an LLM for answer generation.

## Objective

The notebook builds a complete RAG system to:

- load documentation content from the Hugging Face docs dataset
- split long documents into smaller overlapping chunks
- generate embeddings for each chunk
- store embeddings in a local Milvus vector database
- retrieve the most relevant chunks for a user query
- generate a grounded answer using an LLM
- evaluate retrieval and answer quality

## Tech stack

- Dataset: m-ric/huggingface_doc
- Embedding model: BAAI/bge-small-en-v1.5
- Vector database: Milvus
- LLM: OpenAI GPT model (configured in the notebook) or alternative Hugging Face models such as Phi-3 / Qwen
- Evaluation: Opik metrics such as AnswerRelevance and Hallucination

## Workflow overview

The notebook is structured into the following stages:

1. Environment setup
2. Loading the Hugging Face documentation dataset
3. Chunking documents into smaller text blocks
4. Generating embeddings for chunks
5. Creating and populating a Milvus collection
6. Implementing semantic retrieval
7. Generating answers using a prompt-based RAG workflow
8. Evaluating retrieval precision/recall and answer quality
9. Reviewing results and diagnosis

## Key notebook components

### 1. Setup

The notebook installs the required Python libraries, including:

- transformers
- sentence-transformers
- datasets
- pymilvus
- torch
- accelerate
- opik
- tqdm
- openai

It also configures environment variables such as:

- HF_TOKEN
- OPIK_API_KEY
- OPENAI_API_KEY

> Note: API keys are placeholders in the notebook and should be replaced with valid values before running the pipeline.

### 2. Data loading

The dataset is loaded using Hugging Face datasets:

```python
from datasets import load_dataset
dataset = load_dataset("m-ric/huggingface_doc", split="train")
```

A subset of the dataset is used for efficiency, with a limit defined in the notebook as:

```python
MAX_DOCS = 500
```

### 3. Chunking

Documents are split into overlapping chunks using the custom function `chunk_document()`.

Key parameters:

- `chunk_size = 1000`
- `chunk_overlap = 200`

This sliding-window approach creates overlapping segments so retrieval can preserve context across boundaries.

### 4. Embeddings

The notebook uses the sentence-transformer model:

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

The function `generate_embeddings()` accepts a list of texts and returns dense vector representations compatible with Milvus.

### 5. Milvus vector store

The notebook creates a local Milvus database and initializes a collection:

```python
MILVUS_DB_PATH = "./hf_docs_milvus.db"
COLLECTION_NAME = "hf_documentation"
```

The collection is configured with:

- embedding dimension from the embedding model
- metric type: `IP` (Inner Product)
- consistency level: `Strong`

### 6. Retrieval

The function `retrieve_documents()` performs the retrieval step:

- embeds the query using the same embedding model
- searches Milvus for the nearest vectors
- returns the top-k most relevant document chunks with metadata such as text, source, and score

Example usage:

```python
retrieved = retrieve_documents(
    query="How do I fine-tune a transformer model?",
    client=milvus_client,
    collection_name=COLLECTION_NAME,
    embedding_model=embedding_model,
    top_k=3,
)
```

### 7. Generation

The notebook builds a prompt that instructs the model to answer strictly from the retrieved context.

The core prompt template is:

```python
PROMPT_TEMPLATE = """You are a question-answering assistant for a specific knowledge base.

Your task is to answer the question using ONLY the information provided in <context>.

STRICT RULES:
1. Use only the information in <context>.
2. Do not use your own prior knowledge.
3. Do not make assumptions or fill in missing information.
4. If the answer cannot be found in <context>, respond exactly:
"I don't have enough information to answer this question."

<context>
{context}
</context>

<question>
{question}
</question>

Answer:"""
```

The `generate_answer()` function uses the retrieved chunks as context and calls the LLM to produce the final answer.

### 8. Evaluation

The project evaluates both retrieval and generation quality.

#### Retrieval metrics

The notebook computes:

- Precision@K
- Recall@K

Example evaluation values:

```python
K_VALUES = [1, 3, 5]
```

#### Generation metrics

It also uses Opik metrics to calculate:

- AnswerRelevance
- Hallucination

Thresholds used in the notebook:

```python
RELEVANCE_THRESHOLD = 0.80
HALLUCINATION_THRESHOLD = 0.20
RETRIEVAL_THRESHOLD = 0.50
```

These thresholds help flag low-quality answers or weak retrieval quality.

## Important implementation notes

- This notebook contains TODO-based assignment tasks for implementing Milvus setup, insertion, retrieval, and generation.
- The code uses a local Milvus Lite database for simplicity.
- The generation step is configured for an OpenAI API-based model, but can be adapted for Hugging Face models as needed.
- The notebook is designed as an educational or assignment workflow, not as a production-ready deployment pipeline.

## Setup instructions

Install the required dependencies:

```bash
pip install -q --upgrade transformers pymilvus sentence-transformers datasets torch accelerate opik tqdm openai
```

If version conflicts occur, the notebook includes an explicit reinstall step for compatible versions:

```bash
pip uninstall -y transformers sentence-transformers torch accelerate
pip install -q --force-reinstall --no-cache-dir torch==2.13.0 accelerate==0.21.0
pip install -q --force-reinstall --no-cache-dir transformers==4.38.2 sentence-transformers==2.2.2
```

## Example flow

A typical end-to-end usage looks like this:

```python
retrieved_docs = retrieve_documents(
    query="What is Gradio used for?",
    client=milvus_client,
    collection_name=COLLECTION_NAME,
    embedding_model=embedding_model,
    top_k=5,
)

result = generate_answer(
    query="What is Gradio used for?",
    retrieved_docs=retrieved_docs,
    openai_client=openai_client,
    llm_model=LLM_MODEL,
    max_new_tokens=256,
)

print(result["answer"])
```

## Expected output

After running the full notebook, you should see:

- embeddings generated for all document chunks
- vector records inserted into Milvus
- relevant chunks returned for a query
- a grounded response generated from the retrieved context
- retrieval and answer evaluation metrics printed for analysis

## Project resources

- Milvus: https://github.com/milvus-io/milvus
- Microsoft Phi-3-mini-4k-instruct: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Microsoft Phi-3.5-mini-instruct: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- Qwen/Qwen2-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct

## Summary

This notebook demonstrates a complete RAG pipeline for documentation search and Q&A. It is a good example of combining:

- document ingestion
- chunking
- embeddings
- vector search
- contextual generation
- evaluation

for answering knowledge-base questions using a retrieval layer grounded in source documents.
