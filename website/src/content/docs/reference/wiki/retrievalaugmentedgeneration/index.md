---
title: "Retrieval Augmented Generation"
description: "All 181 public types in the AiDotNet.retrievalaugmentedgeneration namespace, organized by kind."
section: "API Reference"
---

**181** public types in this namespace, organized by kind.

## Models & Types (143)

| Type | Summary |
|:-----|:--------|
| [`AgenticChunker`](/docs/reference/wiki/retrievalaugmentedgeneration/agenticchunker/) | Production-ready intelligent chunker that decides where to split text based on semantic boundaries. |
| [`AggregateStats`](/docs/reference/wiki/retrievalaugmentedgeneration/aggregatestats/) | Represents aggregated statistics across multiple evaluations. |
| [`AnswerCorrectnessMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/answercorrectnessmetric/) | Evaluates the factual correctness of generated answers. |
| [`AnswerSimilarityMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/answersimilaritymetric/) | Evaluates the similarity between the generated answer and ground truth. |
| [`AutoCompressor<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/autocompressor/) | Auto-compressor using rule-based text compression for document content reduction. |
| [`AzureSearchDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/azuresearchdocumentstore/) | Azure Cognitive Search-inspired document store with field-based indexing and search capabilities. |
| [`BM25Retriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/bm25retriever/) | BM25 (Best Matching 25) retrieval algorithm for sparse keyword-based search. |
| [`BTreeIndex`](/docs/reference/wiki/retrievalaugmentedgeneration/btreeindex/) | Simple file-based index for mapping string keys to file offsets. |
| [`ChainOfThoughtRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/chainofthoughtretriever/) | Chain-of-Thought retriever that generates reasoning steps before retrieving documents. |
| [`ChromaDBDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/chromadbdocumentstore/) | ChromaDB-based document store designed for simplicity and developer experience. |
| [`CodeAwareTextSplitter`](/docs/reference/wiki/retrievalaugmentedgeneration/codeawaretextsplitter/) | Code-aware text splitter that respects code structure and syntax. |
| [`CohereEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/cohereembeddingmodel/) | Cohere embedding model integration for high-performance embeddings. |
| [`CohereReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/coherereranker/) |  |
| [`ColBERTRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/colbertretriever/) | Retrieves documents using ColBERT's token-level late interaction mechanism (Khattab & Zaharia 2020, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"). |
| [`CommunityIndex<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/communityindex/) | Index structure that maps hierarchy levels and community IDs to their summaries, enabling efficient community-based retrieval for GraphRAG. |
| [`CommunitySummarizer<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/communitysummarizer/) | Generates structured summaries for detected communities in a knowledge graph. |
| [`CommunitySummary`](/docs/reference/wiki/retrievalaugmentedgeneration/communitysummary/) | Represents a summary of a detected community within a knowledge graph. |
| [`ComplExEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/complexembedding/) | ComplEx embedding model: uses complex-valued embeddings with Hermitian dot product scoring. |
| [`ContextCoverageMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/contextcoveragemetric/) | Evaluates how well the retrieved documents cover the information needed to answer the query. |
| [`ContextRelevanceMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/contextrelevancemetric/) | Evaluates the relevance of retrieved context to the query. |
| [`CosineSimilarityMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/cosinesimilaritymetric/) | Cosine similarity metric for vector search. |
| [`CrossEncoderReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/crossencoderreranker/) | Reranks documents using a cross-encoder model that computes fine-grained relevance scores. |
| [`DenseRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/denseretriever/) | Dense retrieval using vector similarity search. |
| [`DistMultEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/distmultembedding/) | DistMult embedding model: bilinear diagonal scoring with Σ(h_k · r_k · t_k). |
| [`DiversityReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/diversityreranker/) | Reranks documents to maximize diversity while maintaining relevance. |
| [`DocumentSummarizer<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/documentsummarizer/) | Document summarizer for creating concise summaries of retrieved content. |
| [`Document<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/document/) | Represents a document with content, metadata, and optional relevance scoring. |
| [`DotProductMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/dotproductmetric/) | Dot product metric for vector search. |
| [`EnhancedGraphRAG<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/enhancedgraphrag/) | Enhanced Graph-based RAG that integrates with `KnowledgeGraph` and supports Local, Global (Leiden community summaries), and DRIFT retrieval modes. |
| [`EuclideanDistanceMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/euclideandistancemetric/) | Euclidean distance metric for vector search. |
| [`EvaluationResult`](/docs/reference/wiki/retrievalaugmentedgeneration/evaluationresult/) | Represents the evaluation results for a single grounded answer. |
| [`ExtractedEntity`](/docs/reference/wiki/retrievalaugmentedgeneration/extractedentity/) | Represents an entity extracted from text during knowledge graph construction. |
| [`ExtractedEntity`](/docs/reference/wiki/retrievalaugmentedgeneration/extractedentity-2/) | Represents an entity extracted from text. |
| [`ExtractedRelation`](/docs/reference/wiki/retrievalaugmentedgeneration/extractedrelation/) | Represents a relation extracted from text between two entities. |
| [`FAISSDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/faissdocumentstore/) |  |
| [`FLARERetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/flareretriever/) | FLARE (Forward-Looking Active REtrieval) pattern that actively decides when and what to retrieve during generation. |
| [`FaithfulnessMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/faithfulnessmetric/) | Evaluates whether the generated answer is faithful to the source documents. |
| [`FileDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/filedocumentstore/) | A durable, file-based vector document store with HNSW indexing and write-ahead logging. |
| [`FileGraphStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/filegraphstore/) | File-based implementation of `IGraphStore` with persistent storage on disk. |
| [`FixedSizeChunkingStrategy`](/docs/reference/wiki/retrievalaugmentedgeneration/fixedsizechunkingstrategy/) | A simple fixed-size text chunking strategy that splits text at character boundaries. |
| [`FlatIndex<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/flatindex/) |  |
| [`GooglePalmEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/googlepalmembeddingmodel/) | Google PaLM embedding model integration via Vertex AI. |
| [`GraphEdge<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphedge/) | Represents a directed edge (relationship) between two nodes in a knowledge graph. |
| [`GraphPath<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphpath/) | Represents a path in the graph: source node -> edge -> target node. |
| [`GraphQueryMatcher<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphquerymatcher/) | Simple pattern matching for graph queries (inspired by Cypher/SPARQL but simplified). |
| [`GraphRAG<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphrag/) | Graph-based RAG (Retrieval Augmented Generation) that combines knowledge graph traversal with vector search for enhanced retrieval. |
| [`GraphRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphretriever/) |  |
| [`GraphTransaction<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphtransaction/) | Transaction coordinator for managing transactions on graph stores with best-effort rollback. |
| [`GroundedAnswer<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/groundedanswer/) | Represents a generated answer with citations and source attribution for transparency and verification. |
| [`HNSWIndex<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/hnswindex/) | Hierarchical Navigable Small World (HNSW) graph-based index for approximate nearest neighbor search. |
| [`HeaderBasedTextSplitter`](/docs/reference/wiki/retrievalaugmentedgeneration/headerbasedtextsplitter/) | Splits structured documents based on header tags (H1, H2, H3, etc.). |
| [`HuggingFaceEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/huggingfaceembeddingmodel/) | HuggingFace-based embedding model for generating embeddings via Inference API. |
| [`HyDEQueryExpansion`](/docs/reference/wiki/retrievalaugmentedgeneration/hydequeryexpansion/) | Hypothetical Document Embeddings (HyDE) query expansion strategy. |
| [`HybridDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/hybriddocumentstore/) |  |
| [`HybridGraphRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/hybridgraphretriever/) | Hybrid retriever that combines vector similarity search with graph traversal for enhanced RAG. |
| [`HybridRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/hybridretriever/) | Hybrid retriever combining dense and sparse retrieval strategies. |
| [`IVFIndex<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/ivfindex/) | Inverted File (IVF) index that partitions the vector space for faster search. |
| [`IdentityQueryProcessor`](/docs/reference/wiki/retrievalaugmentedgeneration/identityqueryprocessor/) | A pass-through query processor that returns the query unchanged. |
| [`IdentityReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/identityreranker/) | A pass-through reranker that returns documents without modifying their order or scores. |
| [`InMemoryDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/inmemorydocumentstore/) |  |
| [`JaccardSimilarityMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/jaccardsimilaritymetric/) | Jaccard similarity metric for vector search. |
| [`KGConstructor<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/kgconstructor/) | Constructs a knowledge graph from unstructured text using heuristic entity and relation extraction. |
| [`KGEmbeddingTrainingResult`](/docs/reference/wiki/retrievalaugmentedgeneration/kgembeddingtrainingresult/) | Contains the results of training a knowledge graph embedding model. |
| [`KeywordExtractionQueryProcessor`](/docs/reference/wiki/retrievalaugmentedgeneration/keywordextractionqueryprocessor/) | Extracts key terms and phrases from queries for focused retrieval. |
| [`KnowledgeGraph<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/knowledgegraph/) | Knowledge graph for storing and querying entity relationships using a pluggable storage backend. |
| [`LLMBasedReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/llmbasedreranker/) | LLM-based reranking using language model relevance assessment. |
| [`LLMContextCompressor<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/llmcontextcompressor/) | LLM-based context compression to reduce token usage while preserving key information. |
| [`LLMQueryExpansion`](/docs/reference/wiki/retrievalaugmentedgeneration/llmqueryexpansion/) | LLM-based query expansion for generating additional query variations. |
| [`LSHIndex<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/lshindex/) | Locality-Sensitive Hashing (LSH) index for approximate nearest neighbor search. |
| [`LearnedSparseEncoderExpansion`](/docs/reference/wiki/retrievalaugmentedgeneration/learnedsparseencoderexpansion/) | Expands queries using learned sparse representations (SPLADE-like) with term importance weighting for hybrid retrieval. |
| [`LeidenCommunityDetector<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/leidencommunitydetector/) | Implements the Leiden algorithm for community detection in knowledge graphs. |
| [`LeidenResult`](/docs/reference/wiki/retrievalaugmentedgeneration/leidenresult/) | Contains the results of Leiden community detection, including hierarchical partitions. |
| [`LemmatizationQueryProcessor`](/docs/reference/wiki/retrievalaugmentedgeneration/lemmatizationqueryprocessor/) | Reduces words to their base or dictionary form (lemma) for better matching. |
| [`LinkPredictionEvaluation`](/docs/reference/wiki/retrievalaugmentedgeneration/linkpredictionevaluation/) | Contains evaluation metrics for a link prediction model. |
| [`LinkPredictor<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/linkpredictor/) | Link prediction engine that uses trained KG embeddings to predict missing triples and evaluate model quality. |
| [`LocalTransformerEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/localtransformerembedding/) | Local transformer embedding model for generating embeddings using ONNX Runtime without external API calls. |
| [`LostInTheMiddleReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/lostinthemiddlereranker/) | Addresses the "lost in the middle" problem by strategically reordering documents. |
| [`ManhattanDistanceMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/manhattandistancemetric/) | Manhattan distance metric for vector search. |
| [`MarkdownTextSplitter`](/docs/reference/wiki/retrievalaugmentedgeneration/markdowntextsplitter/) | Markdown-aware text splitter that respects markdown structure. |
| [`MaximalMarginalRelevanceReranker<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/maximalmarginalrelevancereranker/) | Implements Maximal Marginal Relevance (MMR) reranking to balance relevance and diversity. |
| [`MemoryGraphStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/memorygraphstore/) | In-memory implementation of `IGraphStore` using dictionaries for fast lookups. |
| [`MilvusDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/milvusdocumentstore/) |  |
| [`MultiModalEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/multimodalembeddingmodel/) |  |
| [`MultiModalTextSplitter`](/docs/reference/wiki/retrievalaugmentedgeneration/multimodaltextsplitter/) | Multi-modal splitter for documents containing both text and images. |
| [`MultiQueryExpansion`](/docs/reference/wiki/retrievalaugmentedgeneration/multiqueryexpansion/) | Expands queries by generating multiple query variations from different perspectives using LLM-based reformulation. |
| [`MultiQueryRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/multiqueryretriever/) | Multi-query retriever that generates multiple query variations and merges results. |
| [`MultiStepReasoningResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/multistepreasoningresult/) | Result of multi-step reasoning retrieval. |
| [`MultiStepReasoningResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/multistepreasoningresult-2/) | Result of multi-step reasoning retrieval. |
| [`MultiStepReasoningRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/multistepreasoningretriever/) | Multi-step reasoning retriever that breaks down complex queries into sequential steps. |
| [`MultiVectorRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/multivectorretriever/) |  |
| [`NamedEntityRecognizer`](/docs/reference/wiki/retrievalaugmentedgeneration/namedentityrecognizer/) | Production-ready Named Entity Recognition model using pattern matching and heuristics. |
| [`NeuralGenerator<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/neuralgenerator/) | Neural network-based text generator for RAG systems using LSTM architecture. |
| [`NoiseRobustnessMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/noiserobustnessmetric/) |  |
| [`ONNXSentenceTransformer<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/onnxsentencetransformer/) | Production-ready sentence transformer for generating semantic embeddings using ONNX Runtime. |
| [`OpenAIEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/openaiembeddingmodel/) | OpenAI embedding model for generating embeddings via OpenAI API. |
| [`ParentDocumentRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/parentdocumentretriever/) |  |
| [`PineconeDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/pineconedocumentstore/) |  |
| [`PostgresVectorDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/postgresvectordocumentstore/) |  |
| [`PredictedTriple`](/docs/reference/wiki/retrievalaugmentedgeneration/predictedtriple/) | Represents a predicted (head, relation, tail) triple with its plausibility score. |
| [`QdrantDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/qdrantdocumentstore/) | Qdrant-inspired document store with collection-based organization and payload filtering. |
| [`QueryExpansionProcessor`](/docs/reference/wiki/retrievalaugmentedgeneration/queryexpansionprocessor/) | Expands queries with synonyms and related terms to improve retrieval recall. |
| [`QueryRewritingProcessor<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/queryrewritingprocessor/) | Rewrites queries for clarity and completeness, especially in conversational contexts. |
| [`RAGEvaluator<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/ragevaluator/) | Evaluates RAG system performance using multiple metrics. |
| [`ReasoningStepResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/reasoningstepresult/) | Represents a single reasoning step in the multi-step process. |
| [`ReasoningStepResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/reasoningstepresult-2/) | Represents a single reasoning step in the multi-step process. |
| [`ReciprocalRankFusion<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/reciprocalrankfusion/) | Reciprocal Rank Fusion for combining multiple ranking lists. |
| [`RecursiveCharacterChunkingStrategy`](/docs/reference/wiki/retrievalaugmentedgeneration/recursivecharacterchunkingstrategy/) | Recursively splits text using a hierarchy of separators to preserve document structure. |
| [`RecursiveCharacterTextSplitter`](/docs/reference/wiki/retrievalaugmentedgeneration/recursivecharactertextsplitter/) | Recursive character-based text splitting that preserves semantic meaning. |
| [`RedisVLDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/redisvldocumentstore/) |  |
| [`RetrievalResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/retrievalresult/) | Represents a retrieval result from the hybrid retriever. |
| [`RotatEEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/rotateembedding/) | RotatE embedding model: models relations as rotations in complex vector space. |
| [`SelectiveContextCompressor<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/selectivecontextcompressor/) | Selective context compressor that picks the most relevant sentences based on the query. |
| [`SelfCorrectingRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/selfcorrectingretriever/) | Self-correcting retriever that iteratively refines answers through critique, error detection, and targeted re-retrieval. |
| [`SemanticChunkingStrategy`](/docs/reference/wiki/retrievalaugmentedgeneration/semanticchunkingstrategy/) | Semantic-based text chunking that uses embeddings to group related content. |
| [`SentenceChunkingStrategy`](/docs/reference/wiki/retrievalaugmentedgeneration/sentencechunkingstrategy/) | Splits text into chunks based on sentence boundaries to preserve semantic coherence. |
| [`SentenceTransformersFineTuner<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/sentencetransformersfinetuner/) | Fine-tuner for sentence transformer embedding models on domain-specific training data using triplet loss. |
| [`SlidingWindowChunkingStrategy`](/docs/reference/wiki/retrievalaugmentedgeneration/slidingwindowchunkingstrategy/) | Sliding window chunking strategy with configurable window size and stride. |
| [`SpellCheckQueryProcessor`](/docs/reference/wiki/retrievalaugmentedgeneration/spellcheckqueryprocessor/) | Processes queries by correcting common spelling errors using a dictionary-based approach. |
| [`StaticWordEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/staticwordembeddingmodel/) | Implements a static word embedding model (e.g., GloVe, Word2Vec, FastText). |
| [`StopWordRemovalQueryProcessor`](/docs/reference/wiki/retrievalaugmentedgeneration/stopwordremovalqueryprocessor/) | Removes common stop words from queries to improve retrieval precision. |
| [`StubEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/stubembeddingmodel/) | A deterministic stub embedding model for testing and development that uses hash-based vector generation. |
| [`StubGenerator<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/stubgenerator/) | A simple stub generator for testing and development that creates template-based answers. |
| [`SubQueryExpansion`](/docs/reference/wiki/retrievalaugmentedgeneration/subqueryexpansion/) | Expands complex queries by decomposing them into simpler, focused sub-queries for parallel retrieval. |
| [`TFIDFRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/tfidfretriever/) | TF-IDF (Term Frequency-Inverse Document Frequency) retrieval strategy with cached statistics. |
| [`TableAwareTextSplitter`](/docs/reference/wiki/retrievalaugmentedgeneration/tableawaretextsplitter/) | Specialized splitter that correctly parses and chunks tabular data from documents. |
| [`TemporalTransEEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/temporaltranseembedding/) | Temporal TransE embedding: extends TransE with time-aware scoring via discretized time bins. |
| [`ToolAugmentedReasoningRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/toolaugmentedreasoningretriever/) | Tool-augmented reasoning retriever that can use external tools during reasoning. |
| [`ToolAugmentedResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/toolaugmentedresult/) | Result of tool-augmented reasoning. |
| [`ToolAugmentedResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/toolaugmentedresult-2/) | Result of tool-augmented reasoning. |
| [`ToolInvocation<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/toolinvocation/) | Represents a tool invocation during reasoning. |
| [`ToolInvocation`](/docs/reference/wiki/retrievalaugmentedgeneration/toolinvocation-2/) | Represents a tool invocation during reasoning. |
| [`TransEEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/transeembedding/) | TransE embedding model: entities and relations are vectors in the same space, with the scoring function d(h, r, t) = \|\|h + r - t\|\|. |
| [`TreeOfThoughtsRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/treeofthoughtsretriever/) | Tree-of-Thoughts retriever that explores multiple reasoning paths in a tree structure. |
| [`ValidationResult`](/docs/reference/wiki/retrievalaugmentedgeneration/validationresult/) | The result of a pipeline validation, containing any errors and warnings. |
| [`VectorDocument<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/vectordocument/) | Represents a document paired with its vector embedding for storage and retrieval. |
| [`VectorRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/vectorretriever/) | A dense vector-based retriever that uses embedding similarity for document retrieval. |
| [`VerifiedReasoningResult<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/verifiedreasoningresult/) | Result of verified reasoning retrieval. |
| [`VerifiedReasoningRetriever<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/verifiedreasoningretriever/) | Verified reasoning retriever that validates each reasoning step with critic models. |
| [`VerifiedReasoningStep<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/verifiedreasoningstep/) | Represents a reasoning step with verification information. |
| [`VoyageAIEmbeddingModel<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/voyageaiembeddingmodel/) | Voyage AI-compatible embedding model using ONNX for high-performance local inference. |
| [`WALEntry`](/docs/reference/wiki/retrievalaugmentedgeneration/walentry/) | Represents a single entry in the Write-Ahead Log. |
| [`WeaviateDocumentStore<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/weaviatedocumentstore/) |  |
| [`WriteAheadLog`](/docs/reference/wiki/retrievalaugmentedgeneration/writeaheadlog/) | Write-Ahead Log (WAL) for ensuring ACID properties and crash recovery. |

## Base Classes (11)

| Type | Summary |
|:-----|:--------|
| [`ChunkingStrategyBase`](/docs/reference/wiki/retrievalaugmentedgeneration/chunkingstrategybase/) | Provides a base implementation for text chunking strategies with common functionality. |
| [`ContextCompressorBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/contextcompressorbase/) | Provides a base implementation for context compressors with common functionality. |
| [`DocumentStoreBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/documentstorebase/) | Provides a base implementation for document stores with common functionality. |
| [`EmbeddingModelBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/embeddingmodelbase/) | Provides a base implementation for embedding models with common functionality. |
| [`GeneratorBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/generatorbase/) | Base class for generator implementations providing common functionality and validation. |
| [`KGEmbeddingBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/kgembeddingbase/) | Abstract base class for knowledge graph embedding models providing shared training infrastructure. |
| [`QueryExpansionBase`](/docs/reference/wiki/retrievalaugmentedgeneration/queryexpansionbase/) | Base class for query expansion strategies. |
| [`QueryProcessorBase`](/docs/reference/wiki/retrievalaugmentedgeneration/queryprocessorbase/) | Base class for query processor implementations with common validation logic. |
| [`RAGMetricBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/ragmetricbase/) | Provides a base implementation for RAG evaluation metrics with common functionality. |
| [`RerankerBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/rerankerbase/) | Provides a base implementation for document rerankers with common functionality. |
| [`RetrieverBase<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/retrieverbase/) | Provides a base implementation for document retrievers with common functionality. |

## Interfaces (5)

| Type | Summary |
|:-----|:--------|
| [`IColBertEmbedder<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/icolbertembedder/) | Token-level embedder contract for ColBERT-style late interaction. |
| [`IKnowledgeGraphEmbedding<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/iknowledgegraphembedding/) | Defines the contract for knowledge graph embedding models that learn vector representations of entities and relations. |
| [`IQueryEmbedder<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/iqueryembedder/) | Single-vector query embedder contract for dense retrievers that need a per-query embedding (DPR-style, Karpukhin et al. |
| [`ISimilarityMetric<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/isimilaritymetric/) | Interface for similarity/distance metrics used in vector search. |
| [`IVectorIndex<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/ivectorindex/) | Interface for vector search indexes. |

## Enums (3)

| Type | Summary |
|:-----|:--------|
| [`RetrievalSource`](/docs/reference/wiki/retrievalaugmentedgeneration/retrievalsource/) | Indicates how a result was retrieved. |
| [`TransactionState`](/docs/reference/wiki/retrievalaugmentedgeneration/transactionstate/) | Represents the state of a transaction. |
| [`WALOperationType`](/docs/reference/wiki/retrievalaugmentedgeneration/waloperationtype/) | Types of operations that can be logged in the WAL. |

## Options & Configuration (14)

| Type | Summary |
|:-----|:--------|
| [`ChunkingConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/chunkingconfig/) | Configuration for chunking strategies. |
| [`ContextCompressionConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/contextcompressionconfig/) | Configuration for context compression strategies. |
| [`DocumentStoreConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/documentstoreconfig/) | Configuration for document store components. |
| [`EmbeddingConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/embeddingconfig/) | Configuration for embedding models. |
| [`FileDocumentStoreOptions`](/docs/reference/wiki/retrievalaugmentedgeneration/filedocumentstoreoptions/) | Configuration options for the file-based document store. |
| [`GraphRAGOptions`](/docs/reference/wiki/retrievalaugmentedgeneration/graphragoptions/) | Configuration options for the enhanced GraphRAG retrieval system. |
| [`KGConstructionOptions`](/docs/reference/wiki/retrievalaugmentedgeneration/kgconstructionoptions/) | Configuration options for automated knowledge graph construction from text. |
| [`KGEmbeddingOptions`](/docs/reference/wiki/retrievalaugmentedgeneration/kgembeddingoptions/) | Configuration options for training knowledge graph embedding models. |
| [`KnowledgeGraphOptions`](/docs/reference/wiki/retrievalaugmentedgeneration/knowledgegraphoptions/) | Configuration options for advanced knowledge graph capabilities including embeddings, community detection, link prediction, temporal queries, and KG construction. |
| [`LeidenOptions`](/docs/reference/wiki/retrievalaugmentedgeneration/leidenoptions/) | Configuration options for the Leiden community detection algorithm. |
| [`QueryExpansionConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/queryexpansionconfig/) | Configuration for query expansion strategies. |
| [`RAGConfiguration<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/ragconfiguration/) | Configuration for RAG pipeline components. |
| [`RerankingConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/rerankingconfig/) | Configuration for reranking strategies. |
| [`RetrievalConfig`](/docs/reference/wiki/retrievalaugmentedgeneration/retrievalconfig/) | Configuration for retrieval strategies. |

## Helpers & Utilities (5)

| Type | Summary |
|:-----|:--------|
| [`GraphAnalytics`](/docs/reference/wiki/retrievalaugmentedgeneration/graphanalytics/) | Provides graph analytics algorithms for analyzing knowledge graphs. |
| [`GraphNode<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/graphnode/) | Represents a node in a knowledge graph, typically an entity extracted from text. |
| [`PipelineValidator`](/docs/reference/wiki/retrievalaugmentedgeneration/pipelinevalidator/) | Validates that a set of components forms a valid AI pipeline at runtime. |
| [`RAGConfigurationBuilder<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/ragconfigurationbuilder/) | Builder for constructing RAG configuration. |
| [`ThoughtNode<T>`](/docs/reference/wiki/retrievalaugmentedgeneration/thoughtnode/) | Represents a node in the Tree-of-Thoughts reasoning tree. |

