using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for chunking strategies.
    /// </summary>
    public class ChunkingConfig
    {
        /// <summary>
        /// Gets or sets the chunking strategy type.
        /// </summary>
        public string Strategy { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the chunk size.
        /// </summary>
        public int ChunkSize { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the chunk overlap.
        /// </summary>
        public int ChunkOverlap { get; set; } = 200;

        /// <summary>
        /// Gets or sets additional parameters for the chunking strategy.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
