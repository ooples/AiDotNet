using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for embedding models.
    /// </summary>
    public class EmbeddingConfig
    {
        /// <summary>
        /// Gets or sets the embedding model type.
        /// </summary>
        public string ModelType { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the path to the embedding model.
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the API key for the embedding service.
        /// </summary>
        /// <remarks>
        /// <b>Security:</b> This value should be protected. Do not log, serialize to disk unencrypted, 
        /// or include in error messages. Consider using secure configuration providers such as
        /// environment variables, Azure Key Vault, or IOptions bound to a secrets provider.
        /// </remarks>
        public string ApiKey { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the embedding dimension.
        /// </summary>
        public int EmbeddingDimension { get; set; } = 768;

        /// <summary>
        /// Gets or sets additional parameters for the embedding model.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
