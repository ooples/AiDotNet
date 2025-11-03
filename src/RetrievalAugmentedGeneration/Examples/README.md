# RAG Framework Examples

This directory contains example code demonstrating how to use the AiDotNet RAG framework.

## Basic RAG Pipeline Example

See the complete example code showing:
- Setting up the RAG pipeline components
- Indexing documents with embeddings
- Running queries with and without filters
- Chunking large documents
- Working with metadata

## Usage

The example demonstrates the complete RAG workflow:

1. **Setup**: Create embedding model, document store, retriever, reranker, generator, and pipeline
2. **Indexing**: Add documents with embeddings to the store
3. **Querying**: Ask questions and get grounded answers with citations
4. **Advanced**: Custom configuration, filtering, and chunking

## Quick Start

```csharp
// 1. Create components
var embeddingModel = new StubEmbeddingModel<double>();
var documentStore = new InMemoryDocumentStore<double>();
var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
var reranker = new IdentityReranker();
var generator = new StubGenerator();
var pipeline = new RagPipeline(retriever, reranker, generator);

// 2. Index documents
foreach (var doc in documents)
{
    var embedding = embeddingModel.Embed(doc.Content);
    documentStore.Add(new VectorDocument<double>(doc, embedding));
}

// 3. Ask questions
var answer = pipeline.Generate("What is photosynthesis?");
Console.WriteLine(answer.Answer);
```

## Production Considerations

For production use:
- Replace `StubEmbeddingModel` with real transformer embeddings (Issue #12)
- Replace `StubGenerator` with actual LLM-based generation
- Use `FAISSDocumentStore` or other vector DB for scale
- Implement proper error handling and logging
- Add monitoring and metrics collection

See `../FUTURE-ISSUES.md` for planned enhancements.
