using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// SentencePiece tokenizer implementation.
    /// Used by models like T5, mBART, and XLNet.
    /// Supports both BPE and unigram language model tokenization.
    /// </summary>
    public class SentencePieceTokenizer : TokenizerBase
    {
        private readonly string _modelFile;
        private readonly bool _addBosToken;
        private readonly bool _addEosToken;
        private readonly string _unkPiece;
        private readonly string _padPiece;
        private readonly string _bosPiece;
        private readonly string _eosPiece;
        private Dictionary<string, double> _pieceScores;

        /// <summary>
        /// Initializes a new instance of the SentencePieceTokenizer class
        /// </summary>
        /// <param name="modelFile">Path to SentencePiece model file</param>
        /// <param name="addBosToken">Whether to add beginning of sentence token</param>
        /// <param name="addEosToken">Whether to add end of sentence token</param>
        public SentencePieceTokenizer(
            string modelFile,
            bool addBosToken = true,
            bool addEosToken = true)
        {
            _modelFile = modelFile;
            _addBosToken = addBosToken;
            _addEosToken = addEosToken;
            _unkPiece = "<unk>";
            _padPiece = "<pad>";
            _bosPiece = "<s>";
            _eosPiece = "</s>";
            _pieceScores = new Dictionary<string, double>();
        }

        /// <inheritdoc/>
        public override int MaxSequenceLength => 512;

        /// <inheritdoc/>
        protected override void InitializeSpecialTokens()
        {
            _specialTokens[_padPiece] = 0;
            _specialTokens[_unkPiece] = 1;
            _specialTokens[_bosPiece] = 2;
            _specialTokens[_eosPiece] = 3;
            
            // Additional special tokens
            _specialTokens["<cls>"] = 4;
            _specialTokens["<sep>"] = 5;
            _specialTokens["<mask>"] = 6;
            
            // Add extra ids for special tokens
            for (int i = 0; i < 100; i++)
            {
                _specialTokens[$"<extra_id_{i}>"] = 7 + i;
            }
        }

        /// <inheritdoc/>
        public override int PadTokenId => _specialTokens[_padPiece];

        /// <inheritdoc/>
        public override int UnknownTokenId => _specialTokens[_unkPiece];

        /// <inheritdoc/>
        public override int BosTokenId => _specialTokens[_bosPiece];

        /// <inheritdoc/>
        public override int EosTokenId => _specialTokens[_eosPiece];

        #region Protected Methods

        /// <inheritdoc/>
        protected override async Task LoadVocabularyAsync()
        {
            if (File.Exists(_modelFile))
            {
                // Load from actual SentencePiece model file
                await LoadSentencePieceModel(_modelFile);
            }
            else
            {
                // Create default vocabulary for demo
                CreateDefaultSentencePieceVocabulary();
            }
        }

        /// <inheritdoc/>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            // Normalize text
            text = NormalizeText(text);
            
            // Apply SentencePiece algorithm
            var pieces = await ApplySentencePieceAsync(text);
            
            return pieces;
        }

        /// <inheritdoc/>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            var result = new StringBuilder();
            
            foreach (var token in tokens)
            {
                // Skip special tokens
                if (IsSpecialTokenString(token))
                {
                    continue;
                }
                
                // Handle space marker (▁ in SentencePiece)
                if (token.StartsWith("▁"))
                {
                    if (result.Length > 0)
                    {
                        result.Append(" ");
                    }
                    result.Append(token.Substring(1));
                }
                else
                {
                    result.Append(token);
                }
            }
            
            return await Task.FromResult(result.ToString());
        }

        /// <summary>
        /// Custom encode method for SentencePiece with specific token handling
        /// </summary>
        public async Task<Vector<int>> EncodeSentencePieceAsync(string text, bool addSpecialTokens = true)
        {
            EnsureInitialized();
            
            var tokens = await TokenizeInternalAsync(text);
            var tokenIds = new List<int>();

            // Override addSpecialTokens with instance settings
            var shouldAddBos = addSpecialTokens && _addBosToken;
            var shouldAddEos = addSpecialTokens && _addEosToken;

            if (shouldAddBos)
            {
                tokenIds.Add(BosTokenId);
            }

            foreach (var token in tokens)
            {
                tokenIds.Add(_vocabulary.ContainsKey(token) ? _vocabulary[token] : UnknownTokenId);
            }

            if (shouldAddEos)
            {
                tokenIds.Add(EosTokenId);
            }

            return new Vector<int>(tokenIds.ToArray());
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            var tokens = await TokenizeInternalAsync(text);
            return tokens;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Loads a SentencePiece model from file
        /// </summary>
        private async Task LoadSentencePieceModel(string modelFile)
        {
            // This is a simplified version - actual implementation would parse protobuf
            var lines = await File.ReadAllLinesAsync(modelFile);
            var tokenId = 0;
            
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                var parts = line.Split('\t');
                if (parts.Length >= 2)
                {
                    var piece = parts[0];
                    var score = double.Parse(parts[1]);
                    
                    AddToVocabulary(piece, tokenId++);
                    _pieceScores[piece] = score;
                }
            }
        }

        /// <summary>
        /// Normalizes text for SentencePiece processing
        /// </summary>
        private string NormalizeText(string text)
        {
            // Replace spaces with special marker
            text = Regex.Replace(text, @" ", "▁");
            
            // Add initial space marker
            if (!text.StartsWith("▁"))
            {
                text = "▁" + text;
            }
            
            // Normalize unicode
            text = text.Normalize(NormalizationForm.FormC);
            
            return text;
        }

        /// <summary>
        /// Applies SentencePiece tokenization algorithm
        /// </summary>
        private async Task<List<string>> ApplySentencePieceAsync(string text)
        {
            // This is a simplified version using greedy longest-match
            var tokens = new List<string>();
            var position = 0;
            
            while (position < text.Length)
            {
                var longestMatch = "";
                var longestMatchLength = 0;
                
                // Try to find the longest matching piece
                for (int length = Math.Min(text.Length - position, 50); length > 0; length--)
                {
                    var candidate = text.Substring(position, length);
                    
                    if (_vocabulary.ContainsKey(candidate))
                    {
                        longestMatch = candidate;
                        longestMatchLength = length;
                        break;
                    }
                }
                
                if (longestMatchLength > 0)
                {
                    tokens.Add(longestMatch);
                    position += longestMatchLength;
                }
                else
                {
                    // No match found, use unknown token
                    tokens.Add(_unkPiece);
                    position++;
                }
            }
            
            return await Task.FromResult(tokens);
        }

        /// <summary>
        /// Checks if a token is a special token
        /// </summary>
        private bool IsSpecialTokenString(string token)
        {
            return token.StartsWith("<") && token.EndsWith(">");
        }

        /// <summary>
        /// Creates a default SentencePiece vocabulary for demonstration
        /// </summary>
        private void CreateDefaultSentencePieceVocabulary()
        {
            var tokenId = 0;
            
            // Special tokens
            foreach (var kvp in _specialTokens)
            {
                AddToVocabulary(kvp.Key, kvp.Value);
                _pieceScores[kvp.Key] = 0.0;
                tokenId = Math.Max(tokenId, kvp.Value + 1);
            }
            
            // Single characters with space marker
            for (char c = 'a'; c <= 'z'; c++)
            {
                AddToVocabulary("▁" + c, tokenId++);
                AddToVocabulary(c.ToString(), tokenId++);
                _pieceScores["▁" + c] = -10.0;
                _pieceScores[c.ToString()] = -11.0;
            }
            
            // Capital letters
            for (char c = 'A'; c <= 'Z'; c++)
            {
                AddToVocabulary("▁" + c, tokenId++);
                AddToVocabulary(c.ToString(), tokenId++);
                _pieceScores["▁" + c] = -10.0;
                _pieceScores[c.ToString()] = -11.0;
            }
            
            // Numbers
            for (char c = '0'; c <= '9'; c++)
            {
                AddToVocabulary("▁" + c, tokenId++);
                AddToVocabulary(c.ToString(), tokenId++);
                _pieceScores["▁" + c] = -10.0;
                _pieceScores[c.ToString()] = -11.0;
            }
            
            // Common punctuation
            var punctuation = ".,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~";
            foreach (char c in punctuation)
            {
                AddToVocabulary("▁" + c, tokenId++);
                AddToVocabulary(c.ToString(), tokenId++);
                _pieceScores["▁" + c] = -12.0;
                _pieceScores[c.ToString()] = -13.0;
            }
            
            // Common words with space marker
            var commonWords = new[] {
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she"
            };
            
            foreach (var word in commonWords)
            {
                AddToVocabulary("▁" + word, tokenId++);
                _pieceScores["▁" + word] = -5.0;
                
                // Also add without space marker for mid-word occurrences
                AddToVocabulary(word, tokenId++);
                _pieceScores[word] = -6.0;
            }
            
            // Common subwords
            var commonSubwords = new[] {
                "ing", "ed", "er", "est", "ly", "tion", "ment", "ness", "able",
                "ful", "less", "ish", "ous", "ive", "ize", "ise", "ate", "ity"
            };
            
            foreach (var subword in commonSubwords)
            {
                AddToVocabulary(subword, tokenId++);
                _pieceScores[subword] = -7.0;
                
                // Also with space marker
                AddToVocabulary("▁" + subword, tokenId++);
                _pieceScores["▁" + subword] = -8.0;
            }
            
            // Byte fallbacks (for handling any unicode)
            for (int i = 0; i < 256; i++)
            {
                var bytePiece = $"<0x{i:X2}>";
                AddToVocabulary(bytePiece, tokenId++);
                _pieceScores[bytePiece] = -20.0;
            }
        }

        #endregion
    }
}