using System.Collections.Generic;

namespace AiDotNet.AutoML.SearchSpace
{
    /// <summary>
    /// Defines the Transformer-based search space for neural architecture search.
    /// Includes attention mechanisms, feed-forward networks, and various architectural choices.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class TransformerSearchSpace<T> : SearchSpaceBase<T>
    {
        public TransformerSearchSpace()
        {
            Operations = new List<string>
            {
                "identity",
                "self_attention",
                "multi_head_attention_4",  // 4 heads
                "multi_head_attention_8",  // 8 heads
                "multi_head_attention_16", // 16 heads
                "feed_forward_2x",         // FFN with 2x hidden dimension
                "feed_forward_4x",         // FFN with 4x hidden dimension
                "layer_norm",
                "glu"  // Gated Linear Unit
            };

            MaxNodes = 24;  // Transformers can have many layers
            InputChannels = 768;  // Common embedding dimension
            OutputChannels = 768;

            // Transformer-specific parameters
            AttentionHeads = new List<int> { 4, 8, 12, 16 };
            HiddenDimensions = new List<int> { 768, 1024, 2048, 3072 };
            FeedForwardMultipliers = new List<int> { 2, 4 };
            DropoutRates = new List<double> { 0.0, 0.1, 0.2, 0.3 };
            UsePreNorm = true;  // Pre-LN vs Post-LN
        }

        /// <summary>
        /// Number of attention heads to search over
        /// </summary>
        public List<int> AttentionHeads { get; set; }

        /// <summary>
        /// Hidden dimensions to consider
        /// </summary>
        public List<int> HiddenDimensions { get; set; }

        /// <summary>
        /// Feed-forward expansion ratios
        /// </summary>
        public List<int> FeedForwardMultipliers { get; set; }

        /// <summary>
        /// Dropout rates to search over
        /// </summary>
        public List<double> DropoutRates { get; set; }

        /// <summary>
        /// Whether to use Pre-LayerNorm (true) or Post-LayerNorm (false)
        /// </summary>
        public bool UsePreNorm { get; set; }
    }
}
