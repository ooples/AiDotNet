# RAG Chatbot Sample

A complete chatbot application powered by AiDotNet's Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Document Ingestion** - Add documents via API or web UI
- **Semantic Search** - Vector similarity search using sentence transformers
- **Source Citations** - Every answer includes relevant source documents
- **Interactive Web UI** - Modern chat interface with document management
- **REST API** - Full API for integration with other systems

## Quick Start

```bash
cd samples/end-to-end/ChatbotWithRAG
dotnet run
```

Then open http://localhost:5000 in your browser.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI / API                         │
├─────────────────────────────────────────────────────────────┤
│                      RAG Pipeline                           │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  Embeddings │   Chunker   │  Retriever  │   Vector Store   │
│  (MiniLM)   │ (Recursive) │   (Dense)   │   (In-Memory)    │
└─────────────┴─────────────┴─────────────┴──────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a message and get RAG response |
| `/api/documents` | POST | Add a new document |
| `/api/documents` | GET | List all documents |
| `/api/documents/{id}` | DELETE | Remove a document |
| `/api/stats` | GET | Get system statistics |

## Example API Usage

### Chat

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What neural networks does AiDotNet support?"}'
```

### Add Document

```bash
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "My Document", "content": "Document content here..."}'
```

## Configuration

The RAG pipeline uses these default settings:

| Setting | Value |
|---------|-------|
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Chunk Size | 512 characters |
| Chunk Overlap | 50 characters |
| Top-K Results | 5 |

## Customization

### Change Embedding Model

```csharp
.WithEmbeddings(new SentenceTransformerEmbeddings<float>("all-mpnet-base-v2"))
```

### Add Reranking

```csharp
.WithReranker(new CrossEncoderReranker<float>(
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    topK: 3))
```

### Use FAISS for Large Scale

```csharp
.WithVectorStore(new FAISSVectorStore<float>(
    dimension: 384,
    indexType: FAISSIndexType.IVFFlat))
```

## Sample Documents

The chatbot comes pre-loaded with documentation about:

- AiDotNet Overview
- PredictionModelBuilder Guide
- Neural Network Architectures
- RAG Components
- Distributed Training

## Requirements

- .NET 8.0 SDK
- AiDotNet NuGet package

## Learn More

- [NLP & RAG Tutorial](https://ooples.github.io/AiDotNet/tutorials/nlp/)
- [RAG API Reference](https://ooples.github.io/AiDotNet/api/AiDotNet.RetrievalAugmentedGeneration/)
- [BasicRAG Sample](/samples/nlp/RAG/BasicRAG/)
- [GraphRAG Sample](/samples/nlp/RAG/GraphRAG/)
