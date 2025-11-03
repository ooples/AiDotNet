using System;
using System.Collections.Generic;
using AiDotNet.Helpers;

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
        /// <param name="numericOperations">The numeric operations for type T.</param>
        public RAGConfigurationBuilder(INumericOperations<T> numericOperations)
        {
            if (numericOperations == null) throw new ArgumentNullException(nameof(numericOperations));

            _config = new RAGConfiguration<T>
            {
                NumericOperations = numericOperations,
                DocumentStore = new RAGConfiguration<T>.DocumentStoreConfig(),
                Chunking = new RAGConfiguration<T>.ChunkingConfig(),
                Embedding = new RAGConfiguration<T>.EmbeddingConfig(),
                Retrieval = new RAGConfiguration<T>.RetrievalConfig(),
                Reranking = new RAGConfiguration<T>.RerankingConfig(),
                QueryExpansion = new RAGConfiguration<T>.QueryExpansionConfig(),
                ContextCompression = new RAGConfiguration<T>.ContextCompressionConfig()
            };
        }

        /// <summary>
        /// Configures the document store.
        /// </summary>
        /// <param name="type">The document store type.</param>
        /// <param name="parameters">Additional parameters for the document store.</param>
        /// <returns>The builder instance.</returns>
        public RAGConfigurationBuilder<T> WithDocumentStore(string type, Dictionary<string, object>? parameters = null)
        {
            _config.DocumentStore.Type = type ?? throw new ArgumentNullException(nameof(type));
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
            _config.Chunking.Strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
            _config.Chunking.ChunkSize = chunkSize > 0 ? chunkSize : throw new ArgumentOutOfRangeException(nameof(chunkSize));
            _config.Chunking.ChunkOverlap = chunkOverlap >= 0 ? chunkOverlap : throw new ArgumentOutOfRangeException(nameof(chunkOverlap));
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
            _config.Embedding.ModelType = modelType ?? throw new ArgumentNullException(nameof(modelType));
            _config.Embedding.ModelPath = modelPath;
            _config.Embedding.ApiKey = apiKey;
            _config.Embedding.EmbeddingDimension = embeddingDimension > 0 ? embeddingDimension : throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
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
            _config.Retrieval.Strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
            _config.Retrieval.TopK = topK > 0 ? topK : throw new ArgumentOutOfRangeException(nameof(topK));
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
            _config.Reranking.Enabled = true;
            _config.Reranking.Strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
            _config.Reranking.TopK = topK > 0 ? topK : throw new ArgumentOutOfRangeException(nameof(topK));
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
            _config.QueryExpansion.Enabled = true;
            _config.QueryExpansion.Strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
            _config.QueryExpansion.NumExpansions = numExpansions > 0 ? numExpansions : throw new ArgumentOutOfRangeException(nameof(numExpansions));
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
            _config.ContextCompression.Enabled = true;
            _config.ContextCompression.Strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
            _config.ContextCompression.CompressionRatio = compressionRatio >= 0 && compressionRatio <= 1
                ? compressionRatio
                : throw new ArgumentOutOfRangeException(nameof(compressionRatio));
            _config.ContextCompression.MaxLength = maxLength > 0 ? maxLength : throw new ArgumentOutOfRangeException(nameof(maxLength));
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
