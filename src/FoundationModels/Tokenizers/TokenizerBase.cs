using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// Abstract base class for tokenizer implementations.
    /// Provides common functionality for all tokenizers.
    /// </summary>
    public abstract class TokenizerBase : ITokenizer
    {
        protected Dictionary<string, int> _vocabulary;
        protected Dictionary<int, string> _reverseVocabulary;
        protected Dictionary<string, int> _specialTokens;
        protected bool _isInitialized;
        protected readonly object _initLock = new object();

        /// <summary>
        /// Initializes a new instance of the TokenizerBase class
        /// </summary>
        protected TokenizerBase()
        {
            _vocabulary = new Dictionary<string, int>();
            _reverseVocabulary = new Dictionary<int, string>();
            _specialTokens = new Dictionary<string, int>();
            _isInitialized = false;
            
            InitializeSpecialTokens();
        }

        #region ITokenizer Implementation

        /// <inheritdoc/>
        public int VocabularySize => _vocabulary.Count;

        /// <inheritdoc/>
        public IReadOnlyDictionary<string, int> SpecialTokens => _specialTokens;

        /// <inheritdoc/>
        public abstract int MaxSequenceLength { get; }

        /// <inheritdoc/>
        public virtual int PadTokenId => _specialTokens.ContainsKey("[PAD]") ? _specialTokens["[PAD]"] : 0;

        /// <inheritdoc/>
        public virtual int UnknownTokenId => _specialTokens.ContainsKey("[UNK]") ? _specialTokens["[UNK]"] : 1;

        /// <inheritdoc/>
        public virtual int BosTokenId => _specialTokens.ContainsKey("[CLS]") ? _specialTokens["[CLS]"] : 2;

        /// <inheritdoc/>
        public virtual int EosTokenId => _specialTokens.ContainsKey("[SEP]") ? _specialTokens["[SEP]"] : 3;

        /// <inheritdoc/>
        public bool IsInitialized => _isInitialized;

        /// <inheritdoc/>
        public async Task InitializeAsync()
        {
            if (_isInitialized) return;

            lock (_initLock)
            {
                if (_isInitialized) return;

                LoadVocabularyAsync().GetAwaiter().GetResult();
                _isInitialized = true;
            }

            await Task.CompletedTask;
        }

        /// <inheritdoc/>
        public async Task<Vector<int>> EncodeAsync(string text, bool addSpecialTokens = true)
        {
            EnsureInitialized();
            
            var tokens = await TokenizeInternalAsync(text);
            var tokenIds = new List<int>();

            if (addSpecialTokens)
            {
                tokenIds.Add(BosTokenId);
            }

            foreach (var token in tokens)
            {
                tokenIds.Add(_vocabulary.ContainsKey(token) ? _vocabulary[token] : UnknownTokenId);
            }

            if (addSpecialTokens)
            {
                tokenIds.Add(EosTokenId);
            }

            return new Vector<int>(tokenIds.ToArray());
        }

        /// <inheritdoc/>
        public async Task<TokenizerOutput> EncodeBatchAsync(
            IReadOnlyList<string> texts,
            int? maxLength = null,
            bool padding = true,
            bool truncation = true)
        {
            EnsureInitialized();
            
            var batchSize = texts.Count;
            var effectiveMaxLength = maxLength ?? MaxSequenceLength;
            var encodedTexts = new List<Vector<int>>();
            var sequenceLengths = new List<int>();

            // Encode each text
            foreach (var text in texts)
            {
                var encoded = await EncodeAsync(text, addSpecialTokens: true);
                
                // Truncate if needed
                if (truncation && encoded.Length > effectiveMaxLength)
                {
                    var truncated = new int[effectiveMaxLength];
                    for (int i = 0; i < effectiveMaxLength - 1; i++)
                    {
                        truncated[i] = encoded[i];
                    }
                    truncated[effectiveMaxLength - 1] = EosTokenId;
                    encoded = new Vector<int>(truncated);
                }
                
                encodedTexts.Add(encoded);
                sequenceLengths.Add(encoded.Length);
            }

            // Find max length for padding
            var maxSeqLength = padding ? effectiveMaxLength : encodedTexts.Max(e => e.Length);

            // Create output matrices
            var inputIds = new Matrix<int>(batchSize, maxSeqLength);
            var attentionMask = new Matrix<int>(batchSize, maxSeqLength);

            // Fill matrices
            for (int i = 0; i < batchSize; i++)
            {
                var encoded = encodedTexts[i];
                var seqLen = encoded.Length;

                // Copy token IDs
                for (int j = 0; j < seqLen && j < maxSeqLength; j++)
                {
                    inputIds[i, j] = encoded[j];
                    attentionMask[i, j] = 1;
                }

                // Pad if needed
                if (padding)
                {
                    for (int j = seqLen; j < maxSeqLength; j++)
                    {
                        inputIds[i, j] = PadTokenId;
                        attentionMask[i, j] = 0;
                    }
                }
            }

            return new TokenizerOutput
            {
                InputIds = inputIds,
                AttentionMask = attentionMask,
                SequenceLengths = new Vector<int>(sequenceLengths.ToArray())
            };
        }

        /// <inheritdoc/>
        public async Task<string> DecodeAsync(Vector<int> tokenIds, bool skipSpecialTokens = true)
        {
            EnsureInitialized();
            
            var tokens = new List<string>();
            
            foreach (var tokenId in tokenIds)
            {
                if (skipSpecialTokens && IsSpecialToken(tokenId))
                {
                    continue;
                }
                
                if (_reverseVocabulary.TryGetValue(tokenId, out var token))
                {
                    tokens.Add(token);
                }
            }

            return await PostProcessTokensAsync(tokens);
        }

        /// <inheritdoc/>
        public async Task<IReadOnlyList<string>> DecodeBatchAsync(Matrix<int> tokenIdsBatch, bool skipSpecialTokens = true)
        {
            var results = new List<string>();
            
            for (int i = 0; i < tokenIdsBatch.Rows; i++)
            {
                var row = new int[tokenIdsBatch.Columns];
                for (int j = 0; j < tokenIdsBatch.Columns; j++)
                {
                    row[j] = tokenIdsBatch[i, j];
                }
                
                var decoded = await DecodeAsync(new Vector<int>(row), skipSpecialTokens);
                results.Add(decoded);
            }
            
            return results;
        }

        /// <inheritdoc/>
        public abstract Task<IReadOnlyList<string>> TokenizeAsync(string text);

        /// <inheritdoc/>
        public virtual async Task<Tensor<double>> GetTokenEmbeddingsAsync(Vector<int> tokenIds)
        {
            // This is a placeholder - actual implementations would load real embeddings
            var embeddingDim = 768; // Standard embedding dimension
            var sequenceLength = tokenIds.Length;
            
            var embeddings = new Tensor<double>(new[] { sequenceLength, embeddingDim });
            
            // Initialize with random values for now
            var random = new Random(42);
            for (int i = 0; i < sequenceLength; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    embeddings[i, j] = random.NextDouble() * 2 - 1;
                }
            }
            
            return await Task.FromResult(embeddings);
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Initializes special tokens used by the tokenizer
        /// </summary>
        protected virtual void InitializeSpecialTokens()
        {
            _specialTokens["[PAD]"] = 0;
            _specialTokens["[UNK]"] = 1;
            _specialTokens["[CLS]"] = 2;
            _specialTokens["[SEP]"] = 3;
            _specialTokens["[MASK]"] = 4;
        }

        /// <summary>
        /// Loads the vocabulary from storage
        /// </summary>
        protected abstract Task LoadVocabularyAsync();

        /// <summary>
        /// Performs the actual tokenization
        /// </summary>
        protected abstract Task<List<string>> TokenizeInternalAsync(string text);

        /// <summary>
        /// Post-processes tokens before returning decoded text
        /// </summary>
        protected abstract Task<string> PostProcessTokensAsync(List<string> tokens);

        /// <summary>
        /// Checks if a token ID is a special token
        /// </summary>
        protected bool IsSpecialToken(int tokenId)
        {
            return _specialTokens.ContainsValue(tokenId);
        }

        /// <summary>
        /// Ensures the tokenizer is initialized
        /// </summary>
        protected void EnsureInitialized()
        {
            if (!_isInitialized)
            {
                throw new InvalidOperationException("Tokenizer must be initialized before use. Call InitializeAsync() first.");
            }
        }

        /// <summary>
        /// Adds a token to the vocabulary
        /// </summary>
        protected void AddToVocabulary(string token, int id)
        {
            _vocabulary[token] = id;
            _reverseVocabulary[id] = token;
        }

        #endregion
    }
}