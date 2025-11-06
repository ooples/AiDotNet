using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for context compression strategies.
    /// </summary>
    public class ContextCompressionConfig
    {
        /// <summary>
        /// Gets or sets whether context compression is enabled.
        /// </summary>
        public bool Enabled { get; set; } = false;

        /// <summary>
        /// Gets or sets the context compression strategy type.
        /// </summary>
        public string Strategy { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the compression ratio.
        /// </summary>
        public double CompressionRatio { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets the maximum length after compression.
        /// </summary>
        public int MaxLength { get; set; } = 500;

        /// <summary>
        /// Gets or sets additional parameters for the compression strategy.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
