namespace AiDotNet.Enums;

/// <summary>
/// Defines the type of an AI pipeline component (Tier 2 metadata).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells you what kind of building block a component is.
/// Unlike models (which learn and predict), components transform, route, or process data
/// as part of a larger AI pipeline.
/// </para>
/// </remarks>
public enum ComponentType
{
    // ──────── RAG Pipeline ────────

    /// <summary>
    /// Retrieves relevant documents or passages from a corpus given a query.
    /// Examples: BM25, DPR, ColBERT, HybridRetriever.
    /// </summary>
    Retriever,

    /// <summary>
    /// Re-scores and reorders retrieved documents for improved relevance.
    /// Examples: CrossEncoderReranker, MonoT5, ColBERTReranker.
    /// </summary>
    Reranker,

    /// <summary>
    /// Splits documents into semantically meaningful chunks for indexing.
    /// Examples: SemanticChunker, RecursiveCharacterChunker, SentenceChunker.
    /// </summary>
    Chunker,

    /// <summary>
    /// Transforms or expands queries before retrieval.
    /// Examples: HyDE, MultiQueryExpander, StepBackPrompting.
    /// </summary>
    QueryProcessor,

    /// <summary>
    /// Generates responses using retrieved context and a language model.
    /// Examples: RAGGenerator, FusionInDecoderGenerator.
    /// </summary>
    Generator,

    /// <summary>
    /// Compresses or filters retrieved context before generation.
    /// Examples: LongContextCompressor, SelectiveContextCompressor.
    /// </summary>
    ContextCompressor,

    /// <summary>
    /// Expands a single query into multiple related queries for broader retrieval.
    /// Examples: MultiQueryExpander, QueryDecomposer.
    /// </summary>
    QueryExpander,

    /// <summary>
    /// Stores and retrieves documents or embeddings for a RAG pipeline.
    /// Examples: InMemoryDocumentStore, ChromaStore, PineconeStore.
    /// </summary>
    DocumentStore,

    /// <summary>
    /// Searches vector embeddings for nearest neighbors.
    /// Examples: FaissIndex, HNSWIndex, AnnoyIndex.
    /// </summary>
    VectorIndex,

    /// <summary>
    /// Named entity recognition components for document processing.
    /// Examples: SpacyNER, TransformerNER.
    /// </summary>
    EntityRecognizer,

    // ──────── Learning Strategies ────────

    /// <summary>
    /// Meta-learning algorithms that learn how to learn.
    /// Examples: MAML, Reptile, ProtoNet, MatchingNet.
    /// </summary>
    MetaLearner,

    /// <summary>
    /// Active learning strategies that select the most informative samples to label.
    /// Examples: BALD, BatchBALD, CoreSet, MarginSampling.
    /// </summary>
    ActiveLearner,

    /// <summary>
    /// Continual learning strategies that prevent catastrophic forgetting.
    /// Examples: EWC, GEM, ExperienceReplay, PackNet.
    /// </summary>
    ContinualLearner,

    // ──────── Training Components ────────

    /// <summary>
    /// Knowledge distillation strategies for model compression.
    /// Examples: SoftTargetDistillation, FeatureDistillation, AttentionTransfer.
    /// </summary>
    DistillationStrategy,

    /// <summary>
    /// Federated learning aggregation strategies.
    /// Examples: FedAvg, FedProx, FedBN, Scaffold.
    /// </summary>
    FederatedAggregator,

    /// <summary>
    /// Federated learning trainers that coordinate distributed training.
    /// Examples: InMemoryFederatedTrainer, BufferedAsyncFederatedTrainer.
    /// </summary>
    FederatedTrainer,

    /// <summary>
    /// Privacy mechanisms for differential privacy and secure computation.
    /// Examples: GaussianDifferentialPrivacy, RdpPrivacyAccountant, SecureAggregation.
    /// </summary>
    PrivacyMechanism,

    /// <summary>
    /// Private set intersection protocols for secure data matching.
    /// Examples: CircuitBasedPsi, ObliviousTransferPsi, FuzzyPsi.
    /// </summary>
    PSIProtocol,

    /// <summary>
    /// Personalization strategies for per-client model adaptation.
    /// Examples: FedCPPersonalization, PFedGatePersonalization, KNNPersonalization.
    /// </summary>
    PersonalizationStrategy,

    /// <summary>
    /// Federated unlearning strategies for selective data removal.
    /// Examples: DiffusiveNoiseUnlearner, GradientAscentUnlearner.
    /// </summary>
    FederatedUnlearner,

    /// <summary>
    /// Data loading and dataset management components.
    /// Examples: InMemoryDataset, StreamingDataLoader, LeafDatasetLoader.
    /// </summary>
    DataLoader,

    /// <summary>
    /// Verification and commitment schemes for secure protocols.
    /// Examples: HashCommitmentScheme, MerkleTreeVerifier.
    /// </summary>
    VerificationScheme,

    /// <summary>
    /// Cryptographic primitives used by secure computation protocols.
    /// Examples: HmacSha256Prg, AES-CTR, SecretSharing.
    /// </summary>
    CryptoPrimitive,

    /// <summary>
    /// Graph partitioning and topology management components.
    /// Examples: FederatedGraphPartitioner, CrossClientEdgeDiscovery.
    /// </summary>
    GraphPartitioner,

    /// <summary>
    /// Benchmark and evaluation utilities.
    /// Examples: VerticalFederatedBenchmark, FederatedEvaluator.
    /// </summary>
    BenchmarkUtility,

    /// <summary>
    /// Transfer learning algorithms for domain adaptation.
    /// Examples: FineTuning, FeatureExtraction, AdversarialDA.
    /// </summary>
    TransferAlgorithm,

    /// <summary>
    /// Domain adaptation components for cross-domain learning.
    /// Examples: DANN, MMD, CORAL.
    /// </summary>
    DomainAdapter,

    // ──────── Preprocessing ────────

    /// <summary>
    /// Data scaling/normalization components.
    /// Examples: StandardScaler, MinMaxScaler, RobustScaler.
    /// </summary>
    Scaler,

    /// <summary>
    /// Categorical or feature encoding components.
    /// Examples: OneHotEncoder, LabelEncoder, TargetEncoder.
    /// </summary>
    Encoder,

    /// <summary>
    /// Dimensionality reduction components.
    /// Examples: PCA, UMAP, tSNE, Autoencoder.
    /// </summary>
    DimensionReducer,

    /// <summary>
    /// Feature selection components.
    /// Examples: MutualInformation, LASSO, TreeImportance.
    /// </summary>
    FeatureSelector,

    /// <summary>
    /// Feature generation/engineering components.
    /// Examples: PolynomialFeatures, InteractionFeatures.
    /// </summary>
    FeatureGenerator,

    // ──────── Optimization ────────

    /// <summary>
    /// Optimization algorithms for training.
    /// Examples: Adam, SGD, AdaGrad, Lion.
    /// </summary>
    Optimizer,

    /// <summary>
    /// Learning rate schedulers.
    /// Examples: CosineAnnealing, StepLR, OneCycleLR.
    /// </summary>
    Scheduler,

    /// <summary>
    /// Regularization techniques.
    /// Examples: L1, L2, Dropout, ElasticNet.
    /// </summary>
    Regularizer,

    // ──────── Evaluation ────────

    /// <summary>
    /// Evaluation metrics and benchmarking components.
    /// Examples: RAGASEvaluator, FaithfulnessScorer, ContextRelevanceScorer.
    /// </summary>
    Evaluator
}
