# RAG Q&A System

A production-ready Retrieval-Augmented Generation (RAG) question-answering system built with modern AI technologies.

## Features

- **Document Upload** - Support for PDF, TXT, and CSV files
- **Smart Retrieval** - Vector-based document search using Qdrant
- **AI-Powered Answers** - LLM responses grounded in your documents
- **Source Transparency** - View original documents used for answers
- **Stream Responses** - Real-time answer generation
- **Quality Evaluation** - RAGAS metrics for answer faithfulness & relevancy
- **REST API** - Complete FastAPI with interactive docs

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **LLM & Embeddings**: OpenAI (GPT-4o-mini, text-embedding-3-small)
- **Vector Database**: Qdrant Cloud
- **RAG Orchestration**: LangChain
- **Evaluation**: RAGAS
- **Deployment**: Docker + AWS App Runner + ECR

## Prerequisites

- Python 3.13+
- OpenAI API key
- Qdrant Cloud account & credentials
- AWS account (for deployment)

## Quick Start

1. **Clone and Setup**
```bash
git clone <repo>
cd rag_qna
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Run Locally**
```bash
python -m app.main
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/query` | Ask a question |
| `POST` | `/query/stream` | Stream answer |
| `POST` | `/query/search` | Search documents |
| `POST` | `/documents/upload` | Upload document |
| `GET` | `/documents/info` | Collection stats |
| `GET` | `/health` | Health check |
| `GET` | `/health/ready` | Readiness check |

## Docker

```bash
# Build
docker build -t rag-qa-system .

# Run
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 rag-qa-system
```

## Deployment

Automated CI/CD pipelines for:
- **Build & Test** - GitHub Actions testing

See `.github/workflows` for pipeline details.

## Example Usage

```bash
# Upload a document
curl -X POST -F "file=@document.pdf" http://localhost:8000/documents/upload

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "include_sources": true,
    "enable_evaluation": true
  }'
```

## Configuration

Key settings in `.env`:
- `OPENAI_API_KEY` - OpenAI credentials
- `QDRANT_URL` & `QDRANT_API_KEY` - Vector DB
- `CHUNK_SIZE` - Document chunk size (default: 1000)
- `RETRIEVAL_K` - Number of docs to retrieve (default: 4)
- `LOG_LEVEL` - Logging verbosity

## Security

- Non-root Docker user
- Environment variable configuration
- CORS enabled for development
- Health checks on deployment
