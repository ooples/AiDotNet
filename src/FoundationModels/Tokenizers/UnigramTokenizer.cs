using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// Unigram tokenizer that uses a probabilistic model to find the best tokenization.
    /// This is the algorithm used by SentencePiece for subword tokenization.
    /// </summary>
    public class UnigramTokenizer : TokenizerBase
    {
        private Dictionary<string, double> _tokenProbabilities;
        private readonly int _vocabSize;
        private readonly double _characterCoverage;
        private readonly int _maxPieceLength;
        private readonly bool _treatWhitespaceAsSuffix;

        /// <inheritdoc/>
        public override int MaxSequenceLength => 512;

        /// <summary>
        /// Initializes a new instance of the UnigramTokenizer class
        /// </summary>
        /// <param name="vocabSize">Target vocabulary size</param>
        /// <param name="characterCoverage">Character coverage for vocabulary building</param>
        /// <param name="maxPieceLength">Maximum length of a token piece</param>
        /// <param name="treatWhitespaceAsSuffix">Whether to treat whitespace as suffix (like SentencePiece)</param>
        public UnigramTokenizer(
            int vocabSize = 8000,
            double characterCoverage = 0.9995,
            int maxPieceLength = 16,
            bool treatWhitespaceAsSuffix = true)
        {
            _vocabSize = vocabSize;
            _characterCoverage = characterCoverage;
            _maxPieceLength = maxPieceLength;
            _treatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
            _tokenProbabilities = new Dictionary<string, double>();
        }

        /// <summary>
        /// Loads or builds the unigram vocabulary
        /// </summary>
        protected override async Task LoadVocabularyAsync()
        {
            _vocabulary.Clear();
            _reverseVocabulary.Clear();
            _tokenProbabilities.Clear();

            int tokenId = 0;

            // Add special tokens
            foreach (var specialToken in _specialTokens)
            {
                _vocabulary[specialToken.Key] = tokenId;
                _reverseVocabulary[tokenId] = specialToken.Key;
                _tokenProbabilities[specialToken.Key] = 1.0; // High probability for special tokens
                tokenId++;
            }

            // In a real implementation, this would load a pre-trained vocabulary
            // For now, we'll create a simple character-based vocabulary with common subwords
            await BuildDefaultVocabulary(tokenId);
        }

        /// <summary>
        /// Builds a default vocabulary with common subwords
        /// </summary>
        private async Task BuildDefaultVocabulary(int startTokenId)
        {
            int tokenId = startTokenId;
            
            // Add individual characters
            for (char c = ' '; c <= '~'; c++)
            {
                string token = _treatWhitespaceAsSuffix && c == ' ' ? "▁" : c.ToString();
                if (!_vocabulary.ContainsKey(token))
                {
                    _vocabulary[token] = tokenId;
                    _reverseVocabulary[tokenId] = token;
                    _tokenProbabilities[token] = Math.Log(1.0 / 100); // Base probability
                    tokenId++;
                }
            }

            // Add common English subwords
            var commonSubwords = new[] {
                "▁the", "▁of", "▁and", "▁to", "▁in", "▁a", "▁is", "▁that", "▁it", "▁for",
                "ing", "ed", "ly", "er", "est", "tion", "sion", "ment", "ness", "ful",
                "able", "ible", "ous", "ive", "ize", "ise", "ate", "ity", "al", "ial",
                "▁I", "▁you", "▁he", "▁she", "▁we", "▁they", "▁me", "▁him", "▁her",
                "▁was", "▁were", "▁been", "▁have", "▁has", "▁had", "▁do", "▁does", "▁did",
                "▁will", "▁would", "▁could", "▁should", "▁may", "▁might", "▁must", "▁can",
                "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
                "un", "re", "in", "im", "dis", "pre", "post", "anti", "non", "sub"
            };

            foreach (var subword in commonSubwords)
            {
                if (tokenId >= _vocabSize) break;
                
                if (!_vocabulary.ContainsKey(subword))
                {
                    _vocabulary[subword] = tokenId;
                    _reverseVocabulary[tokenId] = subword;
                    // Higher probability for common subwords
                    _tokenProbabilities[subword] = Math.Log(10.0 / commonSubwords.Length);
                    tokenId++;
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Tokenizes text using the Viterbi algorithm to find the most likely segmentation
        /// </summary>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            if (_treatWhitespaceAsSuffix)
            {
                // Replace spaces with special symbol
                text = text.Replace(" ", "▁");
                if (!text.StartsWith("▁"))
                {
                    text = "▁" + text;
                }
            }

            // Use Viterbi algorithm to find best segmentation
            var bestSegmentation = ViterbiSegmentation(text);
            
            return await Task.FromResult(bestSegmentation);
        }

        /// <summary>
        /// Implements the Viterbi algorithm for finding the best tokenization
        /// </summary>
        private List<string> ViterbiSegmentation(string text)
        {
            int n = text.Length;
            var dp = new double[n + 1];
            var path = new int[n + 1];
            
            // Initialize
            dp[0] = 0;
            for (int i = 1; i <= n; i++)
            {
                dp[i] = double.NegativeInfinity;
            }

            // Dynamic programming
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j <= Math.Min(i + _maxPieceLength, n); j++)
                {
                    string piece = text.Substring(i, j - i);
                    
                    if (_vocabulary.ContainsKey(piece))
                    {
                        double score = dp[i] + GetTokenScore(piece);
                        if (score > dp[j])
                        {
                            dp[j] = score;
                            path[j] = i;
                        }
                    }
                }
            }

            // Backtrack to find segmentation
            var segments = new List<string>();
            int pos = n;
            
            while (pos > 0)
            {
                int start = path[pos];
                segments.Add(text.Substring(start, pos - start));
                pos = start;
            }

            segments.Reverse();
            return segments;
        }

        /// <summary>
        /// Gets the score (log probability) for a token
        /// </summary>
        private double GetTokenScore(string token)
        {
            if (_tokenProbabilities.TryGetValue(token, out double logProb))
            {
                return logProb;
            }
            
            // Unknown token penalty
            return Math.Log(1e-10);
        }

        /// <summary>
        /// Post-processes tokens back to text
        /// </summary>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            var sb = new StringBuilder();
            
            foreach (var token in tokens)
            {
                var processedToken = token;
                
                if (_treatWhitespaceAsSuffix)
                {
                    // Replace special space symbol with actual space
                    processedToken = processedToken.Replace("▁", " ");
                }
                
                sb.Append(processedToken);
            }

            string result = sb.ToString();
            
            // Clean up double spaces and leading space
            if (_treatWhitespaceAsSuffix)
            {
                result = result.Trim();
                while (result.Contains("  "))
                {
                    result = result.Replace("  ", " ");
                }
            }

            return await Task.FromResult(result);
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            return await TokenizeInternalAsync(text);
        }

        /// <summary>
        /// Trains the unigram model on a corpus (simplified version)
        /// </summary>
        public async Task TrainAsync(IEnumerable<string> corpus, int maxIterations = 10)
        {
            // Initialize with all possible substrings
            var initialVocab = new Dictionary<string, int>();
            
            foreach (var text in corpus)
            {
                string processedText = _treatWhitespaceAsSuffix ? text.Replace(" ", "▁") : text;
                
                // Extract all substrings up to maxPieceLength
                for (int i = 0; i < processedText.Length; i++)
                {
                    for (int len = 1; len <= Math.Min(_maxPieceLength, processedText.Length - i); len++)
                    {
                        string substr = processedText.Substring(i, len);
                        initialVocab[substr] = initialVocab.ContainsKey(substr) ? initialVocab[substr] + 1 : 1;
                    }
                }
            }

            // Sort by frequency and take top vocabSize
            var sortedVocab = initialVocab
                .OrderByDescending(kvp => kvp.Value)
                .Take(_vocabSize)
                .ToList();

            // Build vocabulary
            _vocabulary.Clear();
            _reverseVocabulary.Clear();
            _tokenProbabilities.Clear();
            
            int tokenId = 0;
            
            // Add special tokens first
            foreach (var specialToken in _specialTokens)
            {
                _vocabulary[specialToken.Key] = tokenId;
                _reverseVocabulary[tokenId] = specialToken.Key;
                _tokenProbabilities[specialToken.Key] = 0; // Log probability
                tokenId++;
            }

            // Add trained vocabulary
            double totalCount = sortedVocab.Sum(kvp => kvp.Value);
            foreach (var kvp in sortedVocab)
            {
                var token = kvp.Key;
                var count = kvp.Value;
                
                if (!_vocabulary.ContainsKey(token))
                {
                    _vocabulary[token] = tokenId;
                    _reverseVocabulary[tokenId] = token;
                    _tokenProbabilities[token] = Math.Log(count / totalCount);
                    tokenId++;
                }
            }

            _isInitialized = true;
            await Task.CompletedTask;
        }
    }
}