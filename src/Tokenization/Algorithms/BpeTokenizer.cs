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
    /// Byte-Pair Encoding (BPE) tokenizer implementation for subword tokenization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BPE is a data compression algorithm adapted for NLP that learns to merge frequent
    /// character sequences into subword units. It's used by GPT, GPT-2, GPT-3, and many
    /// other modern language models.
    /// </para>
    /// <para><b>For Beginners:</b> BPE is like learning common letter combinations. Imagine
    /// you're creating shorthand notes:
    ///
    /// 1. Start with individual letters: "t", "h", "e", " ", "c", "a", "t"
    /// 2. Notice "th" appears often, so create a symbol for it: "th", "e", " ", ...
    /// 3. Notice "the" appears often, merge again: "the", " ", "cat"
    /// 4. Keep merging until you have a good vocabulary size
    ///
    /// This way, common words like "the" become single tokens, while rare words like
    /// "cryptocurrency" might be split into "crypt" + "ocurrency" or similar subwords.
    ///
    /// Benefits:
    /// - No out-of-vocabulary words (any text can be tokenized)
    /// - Common words are single tokens (efficient)
    /// - Rare words are split into meaningful subwords (handles new words)
    ///
    /// Example tokenization of "unhappiness":
    /// - Full word not in vocabulary, so split into subwords
    /// - Possible result: ["un", "happiness"] or ["un", "happy", "ness"]
    /// </para>
    /// </remarks>
    public class BpeTokenizer : TokenizerBase
    {
        private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
        private readonly Dictionary<(string, string), int> _bpeMerges;
        private readonly Dictionary<string, List<string>> _cache;
        private readonly Regex _patternRegex;

        /// <summary>
        /// Creates a new BPE tokenizer with the specified vocabulary and merge rules.
        /// </summary>
        /// <param name="vocabulary">The vocabulary containing all valid tokens.</param>
        /// <param name="merges">The BPE merges (pairs of tokens to merge and their priority order).</param>
        /// <param name="specialTokens">The special tokens configuration. Defaults to GPT-style tokens.</param>
        /// <param name="pattern">The regex pattern for pre-tokenization. Defaults to GPT-2 pattern.</param>
        /// <remarks>
        /// <para><b>For Beginners:</b> Most users should use the Train method or load a pretrained
        /// tokenizer instead of calling this constructor directly. The merges dictionary contains
        /// rules like ("t", "h") -> 0 meaning "merge t and h first" (lower number = higher priority).
        /// </para>
        /// </remarks>
        public BpeTokenizer(
            IVocabulary vocabulary,
            Dictionary<(string, string), int> merges,
            SpecialTokens? specialTokens = null,
            string? pattern = null)
            : base(vocabulary, specialTokens ?? SpecialTokens.Gpt())
        {
            _bpeMerges = merges ?? throw new ArgumentNullException(nameof(merges));
            _cache = new Dictionary<string, List<string>>();

            // Default GPT-2 pattern for pre-tokenization
            pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            _patternRegex = new Regex(pattern, RegexOptions.Compiled, RegexTimeout);
        }

        /// <summary>
        /// Trains a BPE tokenizer from a text corpus by learning merge rules.
        /// </summary>
        /// <param name="corpus">The training corpus - a collection of text strings.</param>
        /// <param name="vocabSize">The desired vocabulary size (number of unique tokens).</param>
        /// <param name="specialTokens">The special tokens configuration. Defaults to GPT-style tokens.</param>
        /// <param name="pattern">The regex pattern for pre-tokenization. Defaults to GPT-2 pattern.</param>
        /// <returns>A trained BPE tokenizer ready to tokenize text.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> Training learns which letter combinations appear most
        /// frequently in your text. For example, if training on English text:
        ///
        /// 1. The algorithm starts with all individual characters as tokens
        /// 2. It counts all adjacent character pairs in the corpus
        /// 3. The most frequent pair (e.g., "t" + "h") becomes a new token "th"
        /// 4. This repeats until reaching the desired vocabulary size
        ///
        /// Larger vocabulary = longer sequences become single tokens = faster inference
        /// but more memory. Typical sizes: 30,000-50,000 tokens.
        /// </para>
        /// </remarks>
        public static BpeTokenizer Train(
            IEnumerable<string> corpus,
            int vocabSize,
            SpecialTokens? specialTokens = null,
            string? pattern = null)
        {
            if (corpus == null)
                throw new ArgumentNullException(nameof(corpus));
            if (vocabSize < 1)
                throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocabulary size must be at least 1.");

            var corpusList = corpus.ToList();
            specialTokens ??= SpecialTokens.Gpt();

            // Step 1: Build character vocabulary
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);

            // Add special tokens first
            foreach (var token in specialTokens.GetAllSpecialTokens())
            {
                vocabulary.AddToken(token);
            }

            // Handle empty corpus - return minimal tokenizer with only special tokens
            if (corpusList.Count == 0)
            {
                var emptyMerges = new Dictionary<(string, string), int>();
                pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
                return new BpeTokenizer(vocabulary, emptyMerges, specialTokens, pattern);
            }

            // Step 2: Pre-tokenize and get word frequencies
            pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            var preTokenRegex = new Regex(pattern, RegexOptions.Compiled, RegexTimeout);

            var wordFreqs = new Dictionary<string, int>();
            foreach (var text in corpusList)
            {
                var words = preTokenRegex.Matches(text)
                    .Cast<Match>()
                    .Select(m => m.Value);

                foreach (var word in words)
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
                    tokens.AddRange(cachedTokens);
                    continue;
                }

                // Apply BPE
                var bpeTokens = BpeEncode(word);
                _cache[word] = bpeTokens;
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
