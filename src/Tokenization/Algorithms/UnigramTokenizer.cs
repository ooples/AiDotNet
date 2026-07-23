using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Validation;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// Unigram Language Model tokenizer using probabilistic segmentation.
    /// </summary>
    public class UnigramTokenizer : TokenizerBase
    {
        private readonly Dictionary<string, double> _tokenScores;
        private readonly int _maxTokenLength;

        /// <summary>
        /// Creates a new unigram tokenizer.
        /// </summary>
        public UnigramTokenizer(
            IVocabulary vocabulary,
            Dictionary<string, double> tokenScores,
            SpecialTokens specialTokens,
            int maxTokenLength = 16)
            : base(vocabulary, specialTokens)
        {
            Guard.NotNull(tokenScores);
            _tokenScores = tokenScores;
            _maxTokenLength = maxTokenLength;
        }

        /// <summary>
        /// Tokenizes text using Viterbi algorithm for optimal segmentation.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var normalizedText = text.Replace(" ", "\u2581");
            return ViterbiSegment(normalizedText);
        }

        private List<string> ViterbiSegment(string text)
        {
            int n = text.Length;
            var best = new (double score, int prev)[n + 1];
            best[0] = (0.0, -1);

            for (int i = 1; i <= n; i++)
            {
                best[i] = (double.NegativeInfinity, -1);

                int maxLen = Math.Min(i, _maxTokenLength);
                for (int len = 1; len <= maxLen; len++)
                {
                    int start = i - len;
                    string token = text.Substring(start, len);

                    double tokenScore = _tokenScores.TryGetValue(token, out double score) ? score : -100.0;
                    double totalScore = best[start].score + tokenScore;

                    if (totalScore > best[i].score)
                        best[i] = (totalScore, start);
                }

                if (double.IsNegativeInfinity(best[i].score))
                    best[i] = (best[i - 1].score - 10.0, i - 1);
            }

            var tokens = new List<string>();
            int pos = n;
            while (pos > 0)
            {
                int prevPos = best[pos].prev;
                string token = text.Substring(prevPos, pos - prevPos);
                tokens.Add(Vocabulary.ContainsToken(token) ? token : SpecialTokens.UnkToken);
                pos = prevPos;
            }

            tokens.Reverse();
            return tokens;
        }

        protected override string CleanupTokens(List<string> tokens)
        {
            return string.Concat(tokens).Replace("\u2581", " ").TrimStart();
        }

        /// <summary>
        /// Trains a unigram tokenizer from a corpus.
        /// </summary>
        public static UnigramTokenizer Train(
            IEnumerable<string> corpus,
            int vocabSize = 8000,
            SpecialTokens? specialTokens = null)
        {
            if (corpus == null)
                throw new ArgumentNullException(nameof(corpus));

            specialTokens ??= SpecialTokens.Default();
            var corpusList = corpus.ToList();

            // Build initial vocabulary from substrings
            var substringCounts = new Dictionary<string, int>();
            var normalizedTexts = corpusList.Select(text => text.Replace(" ", "\u2581"));
            foreach (var normalized in normalizedTexts)
            {
                for (int i = 0; i < normalized.Length; i++)
                {
                    for (int len = 1; len <= Math.Min(16, normalized.Length - i); len++)
                    {
                        string sub = normalized.Substring(i, len);
                        if (!substringCounts.ContainsKey(sub))
                            substringCounts[sub] = 0;
                        substringCounts[sub]++;
                    }
                }
            }

            // Take top substrings
            var topSubstrings = substringCounts
                .OrderByDescending(kvp => kvp.Value)
                .Take(vocabSize)
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            // Convert counts to log probabilities
            double total = topSubstrings.Values.Sum();
            if (total < double.Epsilon)
                throw new InvalidOperationException("Cannot compute log probabilities: total count is zero.");

            var tokenScores = topSubstrings.ToDictionary(
                kvp => kvp.Key,
                kvp => Math.Log(kvp.Value / total));

            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());
            vocabulary.AddTokens(tokenScores.Keys);

            return new UnigramTokenizer(vocabulary, tokenScores, specialTokens);
        }
    }
}
