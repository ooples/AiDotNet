using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for reranking strategies.
    /// </summary>
    public class RerankingConfig
    {
        /// <summary>
        /// Gets or sets whether reranking is enabled.
        /// </summary>
        public bool Enabled { get; set; } = false;

        /// <summary>
        /// Gets or sets the reranking strategy type.
        /// </summary>
        public string Strategy { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the number of top results after reranking.
        /// </summary>
        public int TopK { get; set; } = 5;

        /// <summary>
        /// Gets or sets additional parameters for the reranking strategy.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
