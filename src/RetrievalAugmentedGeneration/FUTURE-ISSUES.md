# Future RAG Framework Enhancements

This document outlines follow-up GitHub issues to be created for extending the RAG framework beyond the MVP (Issue #284).

## Issue Templates

### 1. Sparse Retrieval Implementations

**Title**: RAG: Add Sparse Retrieval Methods (BM25, TF-IDF, ElasticSearch)

**Description**:
Implement sparse (keyword-based) retrieval methods to complement dense vector retrieval.

**Tasks**:
- [ ] Implement `BM25Retriever` using Okapi BM25 algorithm
- [ ] Implement `TFIDFRetriever` using TF-IDF scoring
- [ ] Implement `ElasticSearchRetriever` for integration with Elasticsearch
- [ ] Add unit tests for each retriever
- [ ] Add performance benchmarks comparing sparse vs dense retrieval
- [ ] Update documentation with usage examples

**Technical Details**:
- Extend `RetrieverBase`
- Implement term frequency and inverse document frequency calculations
- Support stopword filtering and stemming
- Provide configurable parameters (k1, b for BM25)

**Dependencies**: Issue #284 (RAG Framework MVP)

---

### 2. Hybrid Retrieval Strategy

**Title**: RAG: Implement Hybrid Retrieval (Dense + Sparse Fusion)

**Description**:
Combine dense vector retrieval with sparse keyword matching for improved results.

**Tasks**:
- [ ] Implement `HybridRetriever` class
- [ ] Support multiple fusion strategies:
  - Reciprocal Rank Fusion (RRF)
  - Linear score combination
  - Learned score fusion
- [ ] Add configurable weights for dense vs sparse
- [ ] Implement rank-based and score-based fusion
- [ ] Add unit tests and integration tests
- [ ] Benchmark against pure dense and pure sparse

**Technical Details**:
- Extend `RetrieverBase`
- Combine results from a `VectorRetriever` and a sparse retriever
- Normalize scores before fusion
- Support metadata filtering in both retrieval modes

**Dependencies**: Issue #284, Sparse Retrieval issue

---

### 3. Advanced Reranking Strategies

**Title**: RAG: Add Advanced Reranking Models (CrossEncoder, ColBERT, LLM-based)

**Description**:
Implement sophisticated reranking methods for improved relevance.

**Tasks**:
- [ ] Implement `CrossEncoderReranker` using BERT/RoBERTa cross-encoders
- [ ] Implement `ColBERTReranker` using late interaction
- [ ] Implement `MonoT5Reranker` using T5-based ranking
- [ ] Implement `LLMReranker` using LLM prompts for relevance scoring
- [ ] Implement `ScoreBasedReranker` for simple score combination
- [ ] Add unit tests for each reranker
- [ ] Benchmark reranking improvements

**Technical Details**:
- Extend `RerankerBase`
- Integrate with transformer models (depends on Issue #12)
- Support batch reranking for efficiency
- Provide configuration for model selection and parameters

**Dependencies**: Issue #284, Issue #12 (Transformer Embeddings)

---

### 4. Vector Database Integrations

**Title**: RAG: Integrate External Vector Databases (FAISS, Milvus, Pinecone, etc.)

**Description**:
Add support for production-scale vector databases.

**Tasks**:
- [ ] Implement `FAISSDocumentStore<T>` using Facebook AI Similarity Search
- [ ] Implement `MilvusDocumentStore<T>` for distributed vector search
- [ ] Implement `PineconeDocumentStore<T>` for managed cloud service
- [ ] Implement `WeaviateDocumentStore<T>` for GraphQL vector DB
- [ ] Implement `QdrantDocumentStore<T>` for high-performance search
- [ ] Implement `ChromaDocumentStore<T>` for embedding database
- [ ] Add configuration for index types (IVF, HNSW, etc.)
- [ ] Add unit tests and integration tests
- [ ] Provide migration tools from InMemoryDocumentStore

**Technical Details**:
- Extend `DocumentStoreBase<T>`
- Support approximate nearest neighbor (ANN) algorithms
- Handle connection pooling and error retry
- Provide async operations where supported
- Support index persistence and loading

**Dependencies**: Issue #284

---

### 5. Advanced Chunking Strategies

**Title**: RAG: Implement Advanced Text Chunking Methods

**Description**:
Add intelligent text splitting strategies beyond simple fixed-size chunking.

**Tasks**:
- [ ] Implement `RecursiveChunkingStrategy` (respects document structure)
- [ ] Implement `SemanticChunkingStrategy` (embedding-based coherence)
- [ ] Implement `SentenceChunkingStrategy` (sentence boundary awareness)
- [ ] Implement `ParagraphChunkingStrategy` (paragraph-based)
- [ ] Implement `MarkdownChunkingStrategy` (markdown-aware splitting)
- [ ] Implement `CodeChunkingStrategy` (AST-based for code)
- [ ] Add unit tests for each strategy
- [ ] Benchmark chunking quality metrics

**Technical Details**:
- Extend `ChunkingStrategyBase`
- Use NLP libraries for sentence detection
- Implement AST parsing for code chunks
- Support configurable coherence thresholds
- Provide metadata preservation across chunks

**Dependencies**: Issue #284

---

### 6. Additional RAG Evaluation Metrics

**Title**: RAG: Implement Comprehensive Evaluation Metrics

**Description**:
Add metrics beyond the MVP (Faithfulness, Similarity, Coverage) for thorough RAG evaluation.

**Tasks**:
- [ ] Implement `ContextPrecisionMetric` (retrieved context quality)
- [ ] Implement `ContextRecallMetric` (all relevant docs retrieved?)
- [ ] Implement `AnswerRelevanceMetric` (answer addresses query?)
- [ ] Implement `BLEUMetric` (n-gram overlap)
- [ ] Implement `ROUGEMetric` (recall-oriented)
- [ ] Implement `BERTScoreMetric` (semantic similarity)
- [ ] Implement `RAGASMetric` (composite RAGAS score)
- [ ] Add evaluation pipeline for batch assessment
- [ ] Create benchmark datasets

**Technical Details**:
- Extend `RAGMetricBase`
- Support both reference-based and reference-free metrics
- Provide aggregation across multiple queries
- Include statistical significance testing

**Dependencies**: Issue #284

---

### 7. Real Embedding Model Integration

**Title**: RAG: Integrate Real Embedding Models from Issue #12

**Description**:
Replace `StubEmbeddingModel` with actual transformer-based embeddings.

**Tasks**:
- [ ] Create `TransformerEmbeddingModel<T>` adapter
- [ ] Support BERT, RoBERTa, DistilBERT models
- [ ] Implement `SentenceTransformerModel<T>` for sentence-transformers
- [ ] Add API-based embedding models (OpenAI, Cohere)
- [ ] Provide model caching and batching
- [ ] Add migration guide from StubEmbeddingModel
- [ ] Update documentation and examples

**Technical Details**:
- Extend `EmbeddingModelBase<T>`
- Leverage transformer implementations from Issue #12
- Support GPU acceleration where available
- Implement efficient batching for throughput
- Handle token limits and truncation

**Dependencies**: Issue #284, Issue #12 (Transformer Embeddings)

---

### 8. RAG Pipeline Enhancements

**Title**: RAG: Add Advanced Pipeline Features

**Description**:
Enhance `RagPipeline` with additional capabilities.

**Tasks**:
- [ ] Implement query expansion/reformulation
- [ ] Add conversation history support (multi-turn RAG)
- [ ] Implement response streaming
- [ ] Add result caching
- [ ] Support multiple retrieval strategies in one pipeline
- [ ] Implement document diversity boosting
- [ ] Add confidence thresholding
- [ ] Create pipeline serialization (save/load)

**Technical Details**:
- Extend `RagPipeline` or create `AdvancedRagPipeline`
- Implement query rewriting strategies
- Support streaming generators
- Add LRU cache for repeated queries
- Provide hooks for custom processing

**Dependencies**: Issue #284

---

### 9. Document Processing Utilities

**Title**: RAG: Add Document Loaders and Preprocessors

**Description**:
Create utilities for loading and preprocessing documents from various sources.

**Tasks**:
- [ ] Implement PDF document loader
- [ ] Implement Word document loader (.docx)
- [ ] Implement HTML/web page loader
- [ ] Implement markdown loader
- [ ] Implement code file loader (with syntax preservation)
- [ ] Add text cleaning and normalization utilities
- [ ] Implement metadata extraction
- [ ] Add batch processing pipeline

**Technical Details**:
- Create `IDocumentLoader` interface
- Support async loading for large files
- Extract metadata (author, date, title)
- Handle encoding detection
- Provide progress reporting for batch operations

**Dependencies**: Issue #284

---

### 10. Async and Streaming Support

**Title**: RAG: Add Full Async/Await and Streaming Support

**Description**:
Convert synchronous operations to async and add streaming capabilities.

**Tasks**:
- [ ] Add async versions of all interface methods
- [ ] Implement streaming retrieval (`IAsyncEnumerable`)
- [ ] Add streaming generation for real-time responses
- [ ] Support cancellation tokens throughout
- [ ] Implement parallel batch operations
- [ ] Add progress reporting for long operations
- [ ] Update all implementations to support async

**Technical Details**:
- Add `*Async` methods to all interfaces
- Use `IAsyncEnumerable<T>` for streaming results
- Support `CancellationToken` for user cancellation
- Implement async document store operations
- Provide async embedding batching

**Dependencies**: Issue #284

---

## Priority Order

Suggested implementation order based on dependencies and impact:

1. **Sparse Retrieval** - Enables hybrid retrieval
2. **Hybrid Retrieval** - Major performance improvement
3. **Real Embedding Integration** (depends on Issue #12) - Production readiness
4. **Advanced Reranking** - Significant quality improvement
5. **Vector Database Integration** - Scalability
6. **Advanced Chunking** - Better semantic preservation
7. **Additional Metrics** - Better evaluation
8. **Pipeline Enhancements** - Advanced use cases
9. **Document Processing** - Ease of use
10. **Async Support** - Performance and UX

## Notes for Implementation

- Each issue should be implemented as a separate PR
- Maintain backward compatibility with MVP
- Include comprehensive unit tests
- Update main documentation
- Add examples to README
- Follow Interface → Base → Concrete pattern
- Include "For Beginners" documentation sections
