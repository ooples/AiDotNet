using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// Byte-Pair Encoding (BPE) tokenizer implementation.
    /// Used by GPT models and other modern language models.
    /// </summary>
    public class BpeTokenizer : TokenizerBase
    {
        private readonly Dictionary<(string, string), int> _bpeMerges;
        private readonly Dictionary<string, string> _cache;
        private readonly Regex _patternRegex;

        /// <summary>
        /// Creates a new BPE tokenizer.
        /// </summary>
        /// <param name="vocabulary">The vocabulary.</param>
        /// <param name="merges">The BPE merges (pairs of tokens to merge and their priority).</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="pattern">The regex pattern for pre-tokenization (default: GPT-2 pattern).</param>
        public BpeTokenizer(
            IVocabulary vocabulary,
            Dictionary<(string, string), int> merges,
            SpecialTokens? specialTokens = null,
            string? pattern = null)
            : base(vocabulary, specialTokens ?? SpecialTokens.Gpt())
        {
            _bpeMerges = merges ?? throw new ArgumentNullException(nameof(merges));
            _cache = new Dictionary<string, string>();

            // Default GPT-2 pattern for pre-tokenization
            pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            _patternRegex = new Regex(pattern, RegexOptions.Compiled);
        }

        /// <summary>
        /// Trains a BPE tokenizer from a corpus.
        /// </summary>
        /// <param name="corpus">The training corpus.</param>
        /// <param name="vocabSize">The desired vocabulary size.</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="pattern">The regex pattern for pre-tokenization.</param>
        /// <returns>A trained BPE tokenizer.</returns>
        public static BpeTokenizer Train(
            IEnumerable<string> corpus,
            int vocabSize,
            SpecialTokens? specialTokens = null,
            string? pattern = null)
        {
            specialTokens ??= SpecialTokens.Gpt();

            // Step 1: Build character vocabulary
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);

            // Add special tokens first
            foreach (var token in specialTokens.GetAllSpecialTokens())
            {
                vocabulary.AddToken(token);
            }

            // Step 2: Pre-tokenize and get word frequencies
            pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            var preTokenRegex = new Regex(pattern, RegexOptions.Compiled);

            var wordFreqs = new Dictionary<string, int>();
            foreach (var text in corpus)
            {
                var matches = preTokenRegex.Matches(text);
                foreach (var word in matches.Cast<Match>().Select(m => m.Value))
                {
                    wordFreqs[word] = wordFreqs.GetValueOrDefault(word, 0) + 1;
                }
            }

            // Step 3: Initialize word representations as character sequences
            var splits = new Dictionary<string, List<string>>();
            foreach (var word in wordFreqs.Keys)
            {
                var charStrings = word.Select(c => c.ToString()).ToList();
                splits[word] = charStrings;

                // Add characters to vocabulary
                foreach (var charStr in charStrings)
                {
                    vocabulary.AddToken(charStr);
                }
            }

            // Step 4: Iteratively merge the most frequent pair
            var merges = new Dictionary<(string, string), int>();
            var mergeOrder = 0;

            while (vocabulary.Size < vocabSize)
            {
                // Count pairs
                var pairFreqs = new Dictionary<(string, string), int>();
                foreach (var (word, split) in splits)
                {
                    var freq = wordFreqs[word];
                    for (int i = 0; i < split.Count - 1; i++)
                    {
                        var pair = (split[i], split[i + 1]);
                        pairFreqs[pair] = pairFreqs.GetValueOrDefault(pair, 0) + freq;
                    }
                }

                if (pairFreqs.Count == 0)
                    break;

                // Find most frequent pair
                var bestPair = pairFreqs.OrderByDescending(p => p.Value).First().Key;

                // Add merge
                merges[bestPair] = mergeOrder++;

                // Add merged token to vocabulary
                var newToken = bestPair.Item1 + bestPair.Item2;
                vocabulary.AddToken(newToken);

                // Update splits
                var newSplits = new Dictionary<string, List<string>>();
                foreach (var (word, split) in splits)
                {
                    var newSplit = new List<string>();
                    int i = 0;
                    while (i < split.Count)
                    {
                        if (i < split.Count - 1 && split[i] == bestPair.Item1 && split[i + 1] == bestPair.Item2)
                        {
                            newSplit.Add(newToken);
                            i += 2;
                        }
                        else
                        {
                            newSplit.Add(split[i]);
                            i++;
                        }
                    }
                    newSplits[word] = newSplit;
                }
                splits = newSplits;
            }

            return new BpeTokenizer(vocabulary, merges, specialTokens, pattern);
        }

        /// <summary>
        /// Tokenizes text into BPE tokens.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var tokens = new List<string>();

            // Pre-tokenize using the pattern
            var matches = _patternRegex.Matches(text);
            foreach (var word in matches.Cast<Match>().Select(m => m.Value))
            {

                // Check cache
                if (_cache.TryGetValue(word, out var cachedTokens))
                {
                    tokens.AddRange(cachedTokens.Split(' '));
                    continue;
                }

                // Apply BPE
                var bpeTokens = BpeEncode(word);
                _cache[word] = string.Join(" ", bpeTokens);
                tokens.AddRange(bpeTokens);
            }

            return tokens;
        }

        /// <summary>
        /// Applies BPE encoding to a word.
        /// </summary>
        private List<string> BpeEncode(string word)
        {
            if (word.Length == 0)
                return new List<string>();

            // Start with character-level tokens
            var tokens = word.Select(c => c.ToString()).ToList();

            while (tokens.Count > 1)
            {
                // Find the best pair to merge
                var bestPair = ((string, string)?)null;
                var bestRank = int.MaxValue;

                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    if (_bpeMerges.TryGetValue(pair, out var rank) && rank < bestRank)
                    {
                        bestPair = pair;
                        bestRank = rank;
                    }
                }

                if (bestPair == null)
                    break;

                // Merge the best pair
                var newTokens = new List<string>();
                int j = 0;
                while (j < tokens.Count)
                {
                    if (j < tokens.Count - 1 && tokens[j] == bestPair.Value.Item1 && tokens[j + 1] == bestPair.Value.Item2)
                    {
                        newTokens.Add(bestPair.Value.Item1 + bestPair.Value.Item2);
                        j += 2;
                    }
                    else
                    {
                        newTokens.Add(tokens[j]);
                        j++;
                    }
                }
                tokens = newTokens;
            }

            return tokens;
        }

        /// <summary>
        /// Cleans up tokens and converts them back to text.
        /// </summary>
        protected override string CleanupTokens(List<string> tokens)
        {
            if (tokens == null || tokens.Count == 0)
                return string.Empty;

            return string.Join("", tokens);
        }
    }
}
