using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for RAG pipeline components.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class RAGConfiguration<T>
    {
        /// <summary>
        /// Gets or sets the document store configuration.
        /// </summary>
        public DocumentStoreConfig DocumentStore { get; set; } = new DocumentStoreConfig();

        /// <summary>
        /// Gets or sets the chunking strategy configuration.
        /// </summary>
        public ChunkingConfig Chunking { get; set; } = new ChunkingConfig();

        /// <summary>
        /// Gets or sets the embedding model configuration.
        /// </summary>
        public EmbeddingConfig Embedding { get; set; } = new EmbeddingConfig();

        /// <summary>
        /// Gets or sets the retrieval strategy configuration.
        /// </summary>
        public RetrievalConfig Retrieval { get; set; } = new RetrievalConfig();

        /// <summary>
        /// Gets or sets the reranking strategy configuration.
        /// </summary>
        public RerankingConfig Reranking { get; set; } = new RerankingConfig();

        /// <summary>
        /// Gets or sets the query expansion configuration.
        /// </summary>
        public QueryExpansionConfig QueryExpansion { get; set; } = new QueryExpansionConfig();

        /// <summary>
        /// Gets or sets the context compression configuration.
        /// </summary>
        public ContextCompressionConfig ContextCompression { get; set; } = new ContextCompressionConfig();
    }
}
