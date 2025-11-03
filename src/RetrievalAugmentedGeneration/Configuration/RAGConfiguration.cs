using System;
using System.Collections.Generic;
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
        public DocumentStoreConfig DocumentStore { get; set; }

        /// <summary>
        /// Gets or sets the chunking strategy configuration.
        /// </summary>
        public ChunkingConfig Chunking { get; set; }

        /// <summary>
        /// Gets or sets the embedding model configuration.
        /// </summary>
        public EmbeddingConfig Embedding { get; set; }

        /// <summary>
        /// Gets or sets the retrieval strategy configuration.
        /// </summary>
        public RetrievalConfig Retrieval { get; set; }

        /// <summary>
        /// Gets or sets the reranking strategy configuration.
        /// </summary>
        public RerankingConfig Reranking { get; set; }

        /// <summary>
        /// Gets or sets the query expansion configuration.
        /// </summary>
        public QueryExpansionConfig QueryExpansion { get; set; }

        /// <summary>
        /// Gets or sets the context compression configuration.
        /// </summary>
        public ContextCompressionConfig ContextCompression { get; set; }

        /// <summary>
        /// Gets or sets the numeric operations for type T.
        /// </summary>
        public INumericOperations<T> NumericOperations { get; set; }

        public class DocumentStoreConfig
        {
            public string Type { get; set; }
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }

        public class ChunkingConfig
        {
            public string Strategy { get; set; }
            public int ChunkSize { get; set; } = 1000;
            public int ChunkOverlap { get; set; } = 200;
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }

        public class EmbeddingConfig
        {
            public string ModelType { get; set; }
            public string ModelPath { get; set; }
            public string ApiKey { get; set; }
            public int EmbeddingDimension { get; set; } = 768;
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }

        public class RetrievalConfig
        {
            public string Strategy { get; set; }
            public int TopK { get; set; } = 10;
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }

        public class RerankingConfig
        {
            public bool Enabled { get; set; } = false;
            public string Strategy { get; set; }
            public int TopK { get; set; } = 5;
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }

        public class QueryExpansionConfig
        {
            public bool Enabled { get; set; } = false;
            public string Strategy { get; set; }
            public int NumExpansions { get; set; } = 3;
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }

        public class ContextCompressionConfig
        {
            public bool Enabled { get; set; } = false;
            public string Strategy { get; set; }
            public double CompressionRatio { get; set; } = 0.5;
            public int MaxLength { get; set; } = 500;
            public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
        }
    }
}
