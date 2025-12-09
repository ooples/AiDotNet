using System.Collections.Generic;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Interfaces
{
    /// <summary>
    /// Interface for text tokenizers.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// Gets the vocabulary.
        /// </summary>
        IVocabulary Vocabulary { get; }

        /// <summary>
        /// Gets the special tokens.
        /// </summary>
        SpecialTokens SpecialTokens { get; }

        /// <summary>
        /// Encodes text into tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="options">Encoding options.</param>
        /// <returns>The tokenization result.</returns>
        TokenizationResult Encode(string text, EncodingOptions? options = null);

        /// <summary>
        /// Encodes multiple texts into tokens.
        /// </summary>
        /// <param name="texts">The texts to encode.</param>
        /// <param name="options">Encoding options.</param>
        /// <returns>The tokenization results.</returns>
        List<TokenizationResult> EncodeBatch(List<string> texts, EncodingOptions? options = null);

        /// <summary>
        /// Decodes token IDs back into text.
        /// </summary>
        /// <param name="tokenIds">The token IDs to decode.</param>
        /// <param name="skipSpecialTokens">Whether to skip special tokens in the output.</param>
        /// <returns>The decoded text.</returns>
        string Decode(List<int> tokenIds, bool skipSpecialTokens = true);

        /// <summary>
        /// Decodes multiple sequences of token IDs back into text.
        /// </summary>
        /// <param name="tokenIdsBatch">The batch of token IDs to decode.</param>
        /// <param name="skipSpecialTokens">Whether to skip special tokens in the output.</param>
        /// <returns>The decoded texts.</returns>
        List<string> DecodeBatch(List<List<int>> tokenIdsBatch, bool skipSpecialTokens = true);

        /// <summary>
        /// Tokenizes text into subword tokens (without converting to IDs).
        /// </summary>
        /// <param name="text">The text to tokenize.</param>
        /// <returns>The list of tokens.</returns>
        List<string> Tokenize(string text);

        /// <summary>
        /// Converts tokens to token IDs.
        /// </summary>
        /// <param name="tokens">The tokens to convert.</param>
        /// <returns>The token IDs.</returns>
        List<int> ConvertTokensToIds(List<string> tokens);

        /// <summary>
        /// Converts token IDs to tokens.
        /// </summary>
        /// <param name="ids">The token IDs to convert.</param>
        /// <returns>The tokens.</returns>
        List<string> ConvertIdsToTokens(List<int> ids);

        /// <summary>
        /// Gets the vocabulary size.
        /// </summary>
        int VocabularySize { get; }
    }
}
