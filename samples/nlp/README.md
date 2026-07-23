# NLP & LLM Samples

This directory contains examples of Natural Language Processing with AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [TextClassification](./TextClassification/) | Classify text documents |
| [Embeddings](./Embeddings/) | Generate text embeddings |
| [RAG/BasicRAG](./RAG/BasicRAG/) | Simple retrieval-augmented generation |
| [RAG/GraphRAG](./RAG/GraphRAG/) | Knowledge graph enhanced RAG |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.RetrievalAugmentedGeneration;

var rag = new RAGPipeline<float>()
    .WithEmbeddings(new SentenceTransformerEmbeddings<float>())
    .WithVectorStore(new InMemoryVectorStore<float>())
    .WithRetriever(new DenseRetriever<float>(topK: 5))
    .Build();

// Add documents
await rag.IndexDocumentsAsync(documents);

// Query
var response = await rag.QueryAsync("What is AiDotNet?");
```

## RAG Components (50+)

- **Embeddings**: Sentence Transformers, OpenAI, Custom
- **Vector Stores**: In-memory, FAISS, Pinecone, Milvus
- **Retrievers**: Dense, Sparse, Hybrid, Multi-hop
- **Rerankers**: Cross-encoder, ColBERT
- **Chunkers**: Sentence, Recursive, Semantic

## Learn More

- [NLP Tutorial](/docs/tutorials/nlp/)
- [RAG Guide](/docs/tutorials/rag/)
- [API Reference](/api/AiDotNet.RetrievalAugmentedGeneration/)
