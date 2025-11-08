using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for query expansion strategies.
    /// </summary>
    public class QueryExpansionConfig
    {
        /// <summary>
        /// Gets or sets whether query expansion is enabled.
        /// </summary>
        public bool Enabled { get; set; } = false;

        /// <summary>
        /// Gets or sets the query expansion strategy type.
        /// </summary>
        public string Strategy { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the number of expansions to generate.
        /// </summary>
        public int NumExpansions { get; set; } = 3;

        /// <summary>
        /// Gets or sets additional parameters for the query expansion strategy.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
