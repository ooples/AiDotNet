using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Validation;

namespace AiDotNet.Tokenization.Models
{
    /// <summary>
    /// Represents the result of tokenizing text, including token IDs, tokens, and attention masks.
    /// </summary>
    public class TokenizationResult
    {
        /// <summary>
        /// Gets or sets the token IDs.
        /// </summary>
        public List<int> TokenIds { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets the actual tokens (subword strings).
        /// </summary>
        public List<string> Tokens { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the attention mask (1 for real tokens, 0 for padding).
        /// </summary>
        public List<int> AttentionMask { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets the token type IDs (for models that support multiple segments).
        /// </summary>
        public List<int> TokenTypeIds { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets the position IDs for positional embeddings.
        /// </summary>
        public List<int> PositionIds { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets character-level offsets for each token.
        /// </summary>
        public List<(int Start, int End)> Offsets { get; set; } = new List<(int, int)>();

        /// <summary>
        /// Gets or sets additional metadata.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// Gets the number of tokens (excluding padding).
        /// </summary>
        public int Length => AttentionMask.Sum();

        /// <summary>
        /// Gets the total number of token IDs (including padding).
        /// </summary>
        public int TotalLength => TokenIds.Count;

        /// <summary>
        /// Creates an empty tokenization result.
        /// </summary>
        public TokenizationResult()
        {
        }

        /// <summary>
        /// Creates a tokenization result with the specified tokens and IDs.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when tokens and tokenIds have different counts.</exception>
        public TokenizationResult(List<string> tokens, List<int> tokenIds)
        {
            Guard.NotNull(tokens);
            Tokens = tokens;
            Guard.NotNull(tokenIds);
            TokenIds = tokenIds;

            if (tokens.Count != tokenIds.Count)
            {
                throw new ArgumentException(
                    $"Tokens count ({tokens.Count}) must match tokenIds count ({tokenIds.Count}).",
                    nameof(tokenIds));
            }

            AttentionMask = Enumerable.Repeat(1, tokens.Count).ToList();
        }
    }
}
