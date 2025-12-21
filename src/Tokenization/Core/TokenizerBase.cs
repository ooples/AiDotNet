using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Core
{
    /// <summary>
    /// Base class for tokenizers providing common functionality.
    /// </summary>
    public abstract class TokenizerBase : ITokenizer
    {
        /// <summary>
        /// Gets the vocabulary.
        /// </summary>
        public IVocabulary Vocabulary { get; protected set; }

        /// <summary>
        /// Gets the special tokens.
        /// </summary>
        public SpecialTokens SpecialTokens { get; protected set; }

        /// <summary>
        /// Gets the vocabulary size.
        /// </summary>
        public int VocabularySize => Vocabulary.Size;

        /// <summary>
        /// Initializes a new instance of the TokenizerBase class.
        /// </summary>
        protected TokenizerBase(IVocabulary vocabulary, SpecialTokens specialTokens)
        {
            Vocabulary = vocabulary ?? throw new ArgumentNullException(nameof(vocabulary));
            SpecialTokens = specialTokens ?? throw new ArgumentNullException(nameof(specialTokens));
        }

        /// <summary>
        /// Encodes text into tokens.
        /// </summary>
        public virtual TokenizationResult Encode(string text, EncodingOptions? options = null)
        {
            if (string.IsNullOrEmpty(text))
                return new TokenizationResult();

            options ??= new EncodingOptions();

            // Tokenize the text
            var tokens = Tokenize(text);

            // Truncate BEFORE adding special tokens to preserve them
            if (options.Truncation && options.MaxLength.HasValue)
            {
                // Reserve space for special tokens if they will be added
                var reservedSpace = options.AddSpecialTokens ? 2 : 0; // [CLS] and [SEP]
                var maxContentLength = options.MaxLength.Value - reservedSpace;

                if (tokens.Count > maxContentLength)
                {
                    tokens = TruncateSequence(tokens, maxContentLength, options.TruncationSide);
                }
            }

            // Add special tokens if requested (after truncation to preserve them)
            if (options.AddSpecialTokens)
            {
                tokens = AddSpecialTokensToSequence(tokens);
            }

            // Convert tokens to IDs
            var tokenIds = ConvertTokensToIds(tokens);

            // Create attention mask
            var attentionMask = Enumerable.Repeat(1, tokenIds.Count).ToList();

            // Pad if necessary
            if (options.Padding && options.MaxLength.HasValue)
            {
                var paddingLength = options.MaxLength.Value - tokenIds.Count;
                if (paddingLength > 0)
                {
                    var padTokenId = Vocabulary.GetTokenId(SpecialTokens.PadToken);
                    var padding = Enumerable.Repeat(padTokenId, paddingLength).ToList();
                    var paddingMask = Enumerable.Repeat(0, paddingLength).ToList();

                    if (options.PaddingSide == "right")
                    {
                        tokenIds.AddRange(padding);
                        tokens.AddRange(Enumerable.Repeat(SpecialTokens.PadToken, paddingLength));
                        attentionMask.AddRange(paddingMask);
                    }
                    else
                    {
                        tokenIds.InsertRange(0, padding);
                        tokens.InsertRange(0, Enumerable.Repeat(SpecialTokens.PadToken, paddingLength));
                        attentionMask.InsertRange(0, paddingMask);
                    }
                }
            }

            var result = new TokenizationResult
            {
                Tokens = tokens,
                TokenIds = tokenIds,
                AttentionMask = options.ReturnAttentionMask ? attentionMask : new List<int>()
            };

            if (options.ReturnTokenTypeIds)
            {
                result.TokenTypeIds = Enumerable.Repeat(0, tokenIds.Count).ToList();
            }

            if (options.ReturnPositionIds)
            {
                result.PositionIds = Enumerable.Range(0, tokenIds.Count).ToList();
            }

            return result;
        }

        /// <summary>
        /// Encodes multiple texts into tokens.
        /// </summary>
        public virtual List<TokenizationResult> EncodeBatch(List<string> texts, EncodingOptions? options = null)
        {
            return texts.Select(text => Encode(text, options)).ToList();
        }

        /// <summary>
        /// Decodes token IDs back into text.
        /// </summary>
        public virtual string Decode(List<int> tokenIds, bool skipSpecialTokens = true)
        {
            if (tokenIds == null || tokenIds.Count == 0)
                return string.Empty;

            var tokens = ConvertIdsToTokens(tokenIds);

            if (skipSpecialTokens)
            {
                var specialTokensList = SpecialTokens.GetAllSpecialTokens();
                tokens = tokens.Where(t => !specialTokensList.Contains(t)).ToList();
            }

            return CleanupTokens(tokens);
        }

        /// <summary>
        /// Decodes multiple sequences of token IDs back into text.
        /// </summary>
        public virtual List<string> DecodeBatch(List<List<int>> tokenIdsBatch, bool skipSpecialTokens = true)
        {
            return tokenIdsBatch.Select(ids => Decode(ids, skipSpecialTokens)).ToList();
        }

        /// <summary>
        /// Tokenizes text into subword tokens (must be implemented by derived classes).
        /// </summary>
        public abstract List<string> Tokenize(string text);

        /// <summary>
        /// Converts tokens to token IDs.
        /// </summary>
        public virtual List<int> ConvertTokensToIds(List<string> tokens)
        {
            return tokens.Select(t => Vocabulary.GetTokenId(t)).ToList();
        }

        /// <summary>
        /// Converts token IDs to tokens.
        /// </summary>
        public virtual List<string> ConvertIdsToTokens(List<int> ids)
        {
            return ids.Select(id => Vocabulary.GetToken(id) ?? SpecialTokens.UnkToken).ToList();
        }

        /// <summary>
        /// Adds special tokens to a sequence.
        /// </summary>
        protected virtual List<string> AddSpecialTokensToSequence(List<string> tokens)
        {
            var result = new List<string>();

            if (!string.IsNullOrEmpty(SpecialTokens.ClsToken))
                result.Add(SpecialTokens.ClsToken);

            result.AddRange(tokens);

            if (!string.IsNullOrEmpty(SpecialTokens.SepToken))
                result.Add(SpecialTokens.SepToken);

            return result;
        }

        /// <summary>
        /// Truncates a sequence to a maximum length.
        /// </summary>
        protected virtual List<string> TruncateSequence(List<string> tokens, int maxLength, string side)
        {
            if (tokens.Count <= maxLength)
                return tokens;

            return side == "left"
                ? tokens.Skip(tokens.Count - maxLength).ToList()

                : tokens.Take(maxLength).ToList();
        }

        /// <summary>
        /// Cleans up tokens and converts them back to text (must be implemented by derived classes).
        /// </summary>
        protected abstract string CleanupTokens(List<string> tokens);
    }
}
