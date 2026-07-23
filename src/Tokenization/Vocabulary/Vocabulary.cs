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
        private int _unkTokenId;
        private readonly string _unkToken;

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
            _unkToken = unkToken;

            // Add unknown token first
            _unkTokenId = AddToken(unkToken);
        }

        /// <summary>
        /// Creates a vocabulary from an existing token-to-ID mapping.
        /// </summary>
        /// <param name="tokenToId">The token-to-ID mapping.</param>
        /// <param name="unkToken">The unknown token.</param>
        /// <exception cref="ArgumentNullException">Thrown if tokenToId is null.</exception>
        public Vocabulary(Dictionary<string, int> tokenToId, string unkToken = "[UNK]")
        {
            if (tokenToId == null)
                throw new ArgumentNullException(nameof(tokenToId));

            _tokenToId = new Dictionary<string, int>(tokenToId);
            _idToToken = tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            _nextId = tokenToId.Count > 0 ? tokenToId.Values.Max() + 1 : 0;
            _unkToken = unkToken;

            // If unkToken is not in the vocabulary, add it
            if (_tokenToId.TryGetValue(unkToken, out var unkId))
            {
                _unkTokenId = unkId;
            }
            else
            {
                _unkTokenId = _nextId++;
                _tokenToId[unkToken] = _unkTokenId;
                _idToToken[_unkTokenId] = unkToken;
            }
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

            if (_tokenToId.TryGetValue(token, out var existingId))
                return existingId;

            var id = _nextId++;
            _tokenToId[token] = id;
            _idToToken[id] = token;
            return id;
        }

        /// <summary>
        /// Adds multiple tokens to the vocabulary.
        /// </summary>
        /// <param name="tokens">The tokens to add.</param>
        /// <exception cref="ArgumentNullException">Thrown if tokens is null.</exception>
        public void AddTokens(IEnumerable<string> tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));

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
        /// Clears the vocabulary and re-adds the unknown token.
        /// </summary>
        public void Clear()
        {
            _tokenToId.Clear();
            _idToToken.Clear();
            _nextId = 0;

            // Re-add the unknown token to maintain consistency
            _unkTokenId = _nextId++;
            _tokenToId[_unkToken] = _unkTokenId;
            _idToToken[_unkTokenId] = _unkToken;
        }
    }
}
