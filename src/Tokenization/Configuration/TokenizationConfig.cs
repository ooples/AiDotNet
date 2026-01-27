using System;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Configuration
{
    /// <summary>
    /// Configuration options for tokenization in the prediction pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tokenization is the process of breaking text into smaller pieces (tokens)
    /// that a machine learning model can understand. Think of it like breaking a sentence into words,
    /// but sometimes words are further broken into subwords for better handling of unknown words.
    /// </remarks>
    public class TokenizationConfig
    {
        /// <summary>
        /// Gets or sets the default encoding options for tokenization.
        /// </summary>
        public EncodingOptions DefaultEncodingOptions { get; set; } = new EncodingOptions();

        /// <summary>
        /// Gets or sets whether to automatically add special tokens (like [CLS], [SEP]) during encoding.
        /// Default is true.
        /// </summary>
        public bool AddSpecialTokens { get; set; } = true;

        /// <summary>
        /// Gets or sets the maximum sequence length for tokenization.
        /// Sequences longer than this will be truncated.
        /// </summary>
        public int? MaxLength { get; set; }

        /// <summary>
        /// Gets or sets whether to pad sequences to the maximum length.
        /// </summary>
        public bool Padding { get; set; }

        /// <summary>
        /// Gets or sets whether to truncate sequences that exceed max length.
        /// </summary>
        public bool Truncation { get; set; }

        /// <summary>
        /// Gets or sets the side on which to pad sequences ("left" or "right").
        /// Default is "right".
        /// </summary>
        public string PaddingSide { get; set; } = "right";

        /// <summary>
        /// Gets or sets the side on which to truncate sequences ("left" or "right").
        /// Default is "right".
        /// </summary>
        public string TruncationSide { get; set; } = "right";

        /// <summary>
        /// Gets or sets whether to return attention masks.
        /// Default is true.
        /// </summary>
        public bool ReturnAttentionMask { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to return token type IDs (for models like BERT with multiple segments).
        /// Default is false.
        /// </summary>
        public bool ReturnTokenTypeIds { get; set; }

        /// <summary>
        /// Gets or sets whether to cache tokenization results for repeated inputs.
        /// Default is false.
        /// </summary>
        public bool EnableCaching { get; set; }

        /// <summary>
        /// Gets or sets whether to use parallel processing for batch tokenization.
        /// Default is true for batches larger than the threshold.
        /// </summary>
        public bool EnableParallelBatchProcessing { get; set; } = true;

        private int _parallelBatchThreshold = 32;

        /// <summary>
        /// Gets or sets the minimum batch size to trigger parallel processing.
        /// Default is 32.
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if value is less than 1.</exception>
        public int ParallelBatchThreshold
        {
            get => _parallelBatchThreshold;
            set
            {
                if (value < 1)
                    throw new ArgumentOutOfRangeException(nameof(value), "Parallel batch threshold must be at least 1.");
                _parallelBatchThreshold = value;
            }
        }

        /// <summary>
        /// Creates encoding options based on this configuration.
        /// </summary>
        internal EncodingOptions ToEncodingOptions()
        {
            return new EncodingOptions
            {
                AddSpecialTokens = AddSpecialTokens,
                MaxLength = MaxLength,
                Padding = Padding,
                Truncation = Truncation,
                PaddingSide = PaddingSide,
                TruncationSide = TruncationSide,
                ReturnAttentionMask = ReturnAttentionMask,
                ReturnTokenTypeIds = ReturnTokenTypeIds
            };
        }

        /// <summary>
        /// Creates a configuration suitable for BERT-style models.
        /// </summary>
        public static TokenizationConfig ForBert(int maxLength = 512)
        {
            return new TokenizationConfig
            {
                MaxLength = maxLength,
                AddSpecialTokens = true,
                Padding = true,
                Truncation = true,
                ReturnAttentionMask = true,
                ReturnTokenTypeIds = true
            };
        }

        /// <summary>
        /// Creates a configuration suitable for GPT-style models.
        /// </summary>
        public static TokenizationConfig ForGpt(int maxLength = 1024)
        {
            return new TokenizationConfig
            {
                MaxLength = maxLength,
                AddSpecialTokens = false,
                Padding = false,
                Truncation = true,
                TruncationSide = "left",
                ReturnAttentionMask = true,
                ReturnTokenTypeIds = false
            };
        }

        /// <summary>
        /// Creates a configuration suitable for code tokenization.
        /// </summary>
        public static TokenizationConfig ForCode(int maxLength = 2048)
        {
            return new TokenizationConfig
            {
                MaxLength = maxLength,
                AddSpecialTokens = true,
                Padding = false,
                Truncation = true,
                ReturnAttentionMask = true,
                ReturnTokenTypeIds = false
            };
        }
    }
}
