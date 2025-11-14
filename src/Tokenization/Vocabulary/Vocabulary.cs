using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Tokenization.Vocabulary
{
    /// <summary>
    /// Manages a vocabulary of tokens and their IDs.
    /// </summary>
    public class Vocabulary : IVocabulary
    {
        private readonly Dictionary<string, int> _tokenToId;
        private readonly Dictionary<int, string> _idToToken;
        private int _nextId;
        private readonly int _unkTokenId;

        /// <summary>
        /// Gets the vocabulary size.
        /// </summary>
        public int Size => _tokenToId.Count;

        /// <summary>
        /// Gets the token-to-ID mapping.
        /// </summary>
        public IReadOnlyDictionary<string, int> TokenToId => _tokenToId;

        /// <summary>
        /// Gets the ID-to-token mapping.
        /// </summary>
        public IReadOnlyDictionary<int, string> IdToToken => _idToToken;

        /// <summary>
        /// Creates a new vocabulary.
        /// </summary>
        /// <param name="unkToken">The unknown token.</param>
        public Vocabulary(string unkToken = "[UNK]")
        {
            _tokenToId = new Dictionary<string, int>();
            _idToToken = new Dictionary<int, string>();
            _nextId = 0;

            // Add unknown token first
            _unkTokenId = AddToken(unkToken);
        }

        /// <summary>
        /// Creates a vocabulary from an existing token-to-ID mapping.
        /// </summary>
        /// <param name="tokenToId">The token-to-ID mapping.</param>
        /// <param name="unkToken">The unknown token.</param>
        public Vocabulary(Dictionary<string, int> tokenToId, string unkToken = "[UNK]")
        {
            _tokenToId = new Dictionary<string, int>(tokenToId);
            _idToToken = tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            _nextId = tokenToId.Values.Max() + 1;
            _unkTokenId = _tokenToId.ContainsKey(unkToken) ? _tokenToId[unkToken] : 0;
        }

        /// <summary>
        /// Adds a token to the vocabulary.
        /// </summary>
        /// <param name="token">The token to add.</param>
        /// <returns>The token ID.</returns>
        public int AddToken(string token)
        {
            if (string.IsNullOrEmpty(token))
                throw new ArgumentException("Token cannot be null or empty.", nameof(token));

            if (_tokenToId.ContainsKey(token))
                return _tokenToId[token];

            var id = _nextId++;
            _tokenToId[token] = id;
            _idToToken[id] = token;
            return id;
        }

        /// <summary>
        /// Adds multiple tokens to the vocabulary.
        /// </summary>
        /// <param name="tokens">The tokens to add.</param>
        public void AddTokens(IEnumerable<string> tokens)
        {
            foreach (var token in tokens)
            {
                AddToken(token);
            }
        }

        /// <summary>
        /// Gets the token ID for a given token.
        /// </summary>
        /// <param name="token">The token.</param>
        /// <returns>The token ID, or the unknown token ID if not found.</returns>
        public int GetTokenId(string token)
        {
            return _tokenToId.TryGetValue(token, out var id) ? id : _unkTokenId;
        }

        /// <summary>
        /// Gets the token for a given token ID.
        /// </summary>
        /// <param name="id">The token ID.</param>
        /// <returns>The token, or null if not found.</returns>
        public string? GetToken(int id)
        {
            return _idToToken.TryGetValue(id, out var token) ? token : null;
        }

        /// <summary>
        /// Checks if a token exists in the vocabulary.
        /// </summary>
        /// <param name="token">The token to check.</param>
        /// <returns>True if the token exists, false otherwise.</returns>
        public bool ContainsToken(string token)
        {
            return _tokenToId.ContainsKey(token);
        }

        /// <summary>
        /// Checks if a token ID exists in the vocabulary.
        /// </summary>
        /// <param name="id">The token ID to check.</param>
        /// <returns>True if the token ID exists, false otherwise.</returns>
        public bool ContainsId(int id)
        {
            return _idToToken.ContainsKey(id);
        }

        /// <summary>
        /// Gets all tokens in the vocabulary.
        /// </summary>
        /// <returns>All tokens.</returns>
        public IEnumerable<string> GetAllTokens()
        {
            return _tokenToId.Keys;
        }

        /// <summary>
        /// Clears the vocabulary.
        /// </summary>
        public void Clear()
        {
            _tokenToId.Clear();
            _idToToken.Clear();
            _nextId = 0;
        }
    }
}
