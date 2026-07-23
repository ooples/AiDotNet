using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for retrieval strategies.
    /// </summary>
    public class RetrievalConfig
    {
        /// <summary>
        /// Gets or sets the retrieval strategy type.
        /// </summary>
        public string Strategy { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the number of top results to retrieve.
        /// </summary>
        public int TopK { get; set; } = 10;

        /// <summary>
        /// Gets or sets additional parameters for the retrieval strategy.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
