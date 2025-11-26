using System.Collections.Generic;

namespace AiDotNet.Tokenization.Interfaces
{
    /// <summary>
    /// Interface for vocabulary management.
    /// </summary>
    public interface IVocabulary
    {
        /// <summary>
        /// Gets the vocabulary size.
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Adds a token to the vocabulary.
        /// </summary>
        /// <param name="token">The token to add.</param>
        /// <returns>The token ID.</returns>
        int AddToken(string token);

        /// <summary>
        /// Adds multiple tokens to the vocabulary.
        /// </summary>
        /// <param name="tokens">The tokens to add.</param>
        void AddTokens(IEnumerable<string> tokens);

        /// <summary>
        /// Gets the token ID for a given token.
        /// </summary>
        /// <param name="token">The token.</param>
        /// <returns>The token ID, or the unknown token ID if not found.</returns>
        int GetTokenId(string token);

        /// <summary>
        /// Gets the token for a given token ID.
        /// </summary>
        /// <param name="id">The token ID.</param>
        /// <returns>The token, or null if not found.</returns>
        string? GetToken(int id);

        /// <summary>
        /// Checks if a token exists in the vocabulary.
        /// </summary>
        /// <param name="token">The token to check.</param>
        /// <returns>True if the token exists, false otherwise.</returns>
        bool ContainsToken(string token);

        /// <summary>
        /// Checks if a token ID exists in the vocabulary.
        /// </summary>
        /// <param name="id">The token ID to check.</param>
        /// <returns>True if the token ID exists, false otherwise.</returns>
        bool ContainsId(int id);

        /// <summary>
        /// Gets all tokens in the vocabulary.
        /// </summary>
        /// <returns>All tokens.</returns>
        IEnumerable<string> GetAllTokens();

        /// <summary>
        /// Gets the token-to-ID mapping.
        /// </summary>
        IReadOnlyDictionary<string, int> TokenToId { get; }

        /// <summary>
        /// Gets the ID-to-token mapping.
        /// </summary>
        IReadOnlyDictionary<int, string> IdToToken { get; }

        /// <summary>
        /// Clears the vocabulary.
        /// </summary>
        void Clear();
    }
}
