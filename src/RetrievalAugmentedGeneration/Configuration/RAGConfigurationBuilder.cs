

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Builder for constructing RAG configuration.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class RAGConfigurationBuilder<T>
    {
        private readonly RAGConfiguration<T> _config;

        /// <summary>
        /// Initializes a new instance of the <see cref="RAGConfigurationBuilder{T}"/> class.
        /// </summary>
        public RAGConfigurationBuilder()
        {
            _config = new RAGConfiguration<T>();
        }

        /// <summary>
        /// Configures the document store.
        /// </summary>
        /// <param name="type">The document store type.</param>
        /// <param name="parameters">Additional parameters for the document store.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithDocumentStore(string type, Dictionary<string, object>? parameters = null)
        {
            if (string.IsNullOrWhiteSpace(type))
                throw new ArgumentException("Document store type cannot be null or empty", nameof(type));

            _config.DocumentStore.Type = type;
            if (parameters != null)
            {
                _config.DocumentStore.Parameters = parameters;
            }
            return this;
        }

        /// <summary>
        /// Configures the chunking strategy.
        /// </summary>
        /// <param name="strategy">The chunking strategy name.</param>
        /// <param name="chunkSize">The chunk size.</param>
        /// <param name="chunkOverlap">The chunk overlap.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithChunking(string strategy, int chunkSize = 1000, int chunkOverlap = 200)
        {
            if (string.IsNullOrWhiteSpace(strategy))
                throw new ArgumentException("Chunking strategy cannot be null or empty", nameof(strategy));
            if (chunkSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(chunkSize), "Chunk size must be greater than zero");
            if (chunkOverlap < 0)
                throw new ArgumentOutOfRangeException(nameof(chunkOverlap), "Chunk overlap cannot be negative");

            _config.Chunking.Strategy = strategy;
            _config.Chunking.ChunkSize = chunkSize;
            _config.Chunking.ChunkOverlap = chunkOverlap;
            return this;
        }

        /// <summary>
        /// Configures the embedding model.
        /// </summary>
        /// <param name="modelType">The embedding model type.</param>
        /// <param name="modelPath">The path to the model.</param>
        /// <param name="apiKey">The API key (if required).</param>
        /// <param name="embeddingDimension">The embedding dimension.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithEmbedding(string modelType, string modelPath = "", string apiKey = "", int embeddingDimension = 768)
        {
            if (string.IsNullOrWhiteSpace(modelType))
                throw new ArgumentException("Model type cannot be null or empty", nameof(modelType));
            if (embeddingDimension <= 0)
                throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "Embedding dimension must be greater than zero");

            _config.Embedding.ModelType = modelType;
            _config.Embedding.ModelPath = modelPath;
            _config.Embedding.ApiKey = apiKey;
            _config.Embedding.EmbeddingDimension = embeddingDimension;
            return this;
        }

        /// <summary>
        /// Configures the retrieval strategy.
        /// </summary>
        /// <param name="strategy">The retrieval strategy name.</param>
        /// <param name="topK">The number of documents to retrieve.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithRetrieval(string strategy, int topK = 10)
        {
            if (string.IsNullOrWhiteSpace(strategy))
                throw new ArgumentException("Retrieval strategy cannot be null or empty", nameof(strategy));
            if (topK <= 0)
                throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be greater than zero");

            _config.Retrieval.Strategy = strategy;
            _config.Retrieval.TopK = topK;
            return this;
        }

        /// <summary>
        /// Configures the reranking strategy.
        /// </summary>
        /// <param name="strategy">The reranking strategy name.</param>
        /// <param name="topK">The number of documents to return after reranking.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithReranking(string strategy, int topK = 5)
        {
            if (string.IsNullOrWhiteSpace(strategy))
                throw new ArgumentException("Reranking strategy cannot be null or empty", nameof(strategy));
            if (topK <= 0)
                throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be greater than zero");

            _config.Reranking.Enabled = true;
            _config.Reranking.Strategy = strategy;
            _config.Reranking.TopK = topK;
            return this;
        }

        /// <summary>
        /// Configures query expansion.
        /// </summary>
        /// <param name="strategy">The query expansion strategy name.</param>
        /// <param name="numExpansions">The number of query expansions to generate.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithQueryExpansion(string strategy, int numExpansions = 3)
        {
            if (string.IsNullOrWhiteSpace(strategy))
                throw new ArgumentException("Query expansion strategy cannot be null or empty", nameof(strategy));
            if (numExpansions <= 0)
                throw new ArgumentOutOfRangeException(nameof(numExpansions), "Number of expansions must be greater than zero");

            _config.QueryExpansion.Enabled = true;
            _config.QueryExpansion.Strategy = strategy;
            _config.QueryExpansion.NumExpansions = numExpansions;
            return this;
        }

        /// <summary>
        /// Configures context compression.
        /// </summary>
        /// <param name="strategy">The compression strategy name.</param>
        /// <param name="compressionRatio">The compression ratio (0.0 to 1.0).</param>
        /// <param name="maxLength">The maximum length of compressed content.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithContextCompression(string strategy, double compressionRatio = 0.5, int maxLength = 500)
        {
            if (string.IsNullOrWhiteSpace(strategy))
                throw new ArgumentException("Compression strategy cannot be null or empty", nameof(strategy));
            if (compressionRatio < 0 || compressionRatio > 1)
                throw new ArgumentOutOfRangeException(nameof(compressionRatio), "Compression ratio must be between 0 and 1");
            if (maxLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxLength), "Max length must be greater than zero");

            _config.ContextCompression.Enabled = true;
            _config.ContextCompression.Strategy = strategy;
            _config.ContextCompression.CompressionRatio = compressionRatio;
            _config.ContextCompression.MaxLength = maxLength;
            return this;
        }

        /// <summary>
        /// Builds the RAG configuration.
        /// </summary>
        /// <returns>The configured RAG configuration.</returns>
        public RAGConfiguration<T> Build()
        {
            ValidateConfiguration();
            return _config;
        }

        private void ValidateConfiguration()
        {
            if (string.IsNullOrEmpty(_config.DocumentStore.Type))
                throw new InvalidOperationException("Document store type must be configured");

            if (string.IsNullOrEmpty(_config.Chunking.Strategy))
                throw new InvalidOperationException("Chunking strategy must be configured");

            if (string.IsNullOrEmpty(_config.Embedding.ModelType))
                throw new InvalidOperationException("Embedding model type must be configured");

            if (string.IsNullOrEmpty(_config.Retrieval.Strategy))
                throw new InvalidOperationException("Retrieval strategy must be configured");
        }
    }
}
