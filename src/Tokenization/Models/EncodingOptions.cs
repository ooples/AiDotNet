namespace AiDotNet.Tokenization.Models
{
    /// <summary>
    /// Options for encoding text into tokens.
    /// </summary>
    public class EncodingOptions
    {
        /// <summary>
        /// Gets or sets whether to add special tokens (e.g., [CLS], [SEP]).
        /// </summary>
        public bool AddSpecialTokens { get; set; } = true;

        /// <summary>
        /// Gets or sets the maximum sequence length. Sequences longer than this will be truncated.
        /// </summary>
        public int? MaxLength { get; set; }

        /// <summary>
        /// Gets or sets whether to pad sequences to MaxLength.
        /// </summary>
        public bool Padding { get; set; } = false;

        /// <summary>
        /// Gets or sets the padding side ("right" or "left").
        /// </summary>
        public string PaddingSide { get; set; } = "right";

        /// <summary>
        /// Gets or sets whether to truncate sequences that exceed MaxLength.
        /// </summary>
        public bool Truncation { get; set; } = false;

        /// <summary>
        /// Gets or sets the truncation side ("right" or "left").
        /// </summary>
        public string TruncationSide { get; set; } = "right";

        /// <summary>
        /// Gets or sets whether to return attention masks.
        /// </summary>
        public bool ReturnAttentionMask { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to return token type IDs.
        /// </summary>
        public bool ReturnTokenTypeIds { get; set; } = false;

        /// <summary>
        /// Gets or sets whether to return character offsets.
        /// </summary>
        public bool ReturnOffsets { get; set; } = false;

        /// <summary>
        /// Gets or sets the stride for overflow handling (used when truncating).
        /// </summary>
        public int Stride { get; set; } = 0;

        /// <summary>
        /// Creates default encoding options.
        /// </summary>
        public EncodingOptions()
        {
        }
    }
}
