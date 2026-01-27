using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// WordPiece tokenizer implementation.
    /// Used by BERT and similar models.
    /// </summary>
    public class WordPieceTokenizer : TokenizerBase
    {
        private readonly string _continuingSubwordPrefix;
        private readonly int _maxInputCharsPerWord;

        /// <summary>
        /// Creates a new WordPiece tokenizer.
        /// </summary>
        /// <param name="vocabulary">The vocabulary.</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="continuingSubwordPrefix">The prefix for continuing subwords (default: "##").</param>
        /// <param name="maxInputCharsPerWord">Maximum characters per word (default: 100).</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if maxInputCharsPerWord is less than 1.</exception>
        public WordPieceTokenizer(
            IVocabulary vocabulary,
            SpecialTokens? specialTokens = null,
            string continuingSubwordPrefix = "##",
            int maxInputCharsPerWord = 100)
            : base(vocabulary, specialTokens ?? SpecialTokens.Bert())
        {
            if (maxInputCharsPerWord < 1)
                throw new ArgumentOutOfRangeException(nameof(maxInputCharsPerWord), "Max input chars per word must be at least 1.");

            _continuingSubwordPrefix = continuingSubwordPrefix;
            _maxInputCharsPerWord = maxInputCharsPerWord;
        }

        /// <summary>
        /// Trains a WordPiece tokenizer from a corpus.
        /// </summary>
        /// <param name="corpus">The training corpus.</param>
        /// <param name="vocabSize">The desired vocabulary size.</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="continuingSubwordPrefix">The prefix for continuing subwords.</param>
        /// <returns>A trained WordPiece tokenizer.</returns>
        /// <exception cref="ArgumentNullException">Thrown if corpus is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if vocabSize is less than 1.</exception>
        public static WordPieceTokenizer Train(
            IEnumerable<string> corpus,
            int vocabSize,
            SpecialTokens? specialTokens = null,
            string continuingSubwordPrefix = "##")
        {
            if (corpus == null)
                throw new ArgumentNullException(nameof(corpus));
            if (vocabSize < 1)
                throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocabulary size must be at least 1.");

            specialTokens ??= SpecialTokens.Bert();

            // Step 1: Build character vocabulary
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);

            // Add special tokens first
            foreach (var token in specialTokens.GetAllSpecialTokens())
            {
                vocabulary.AddToken(token);
            }

            // Step 2: Pre-tokenize and get word frequencies
            var wordFreqs = new Dictionary<string, int>();
            foreach (var text in corpus)
            {
                var words = text.ToLowerInvariant()
                    .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                foreach (var word in words)
                {
                    // Remove punctuation for basic training
                    var cleanWord = new string(word.Where(c => char.IsLetterOrDigit(c)).ToArray());
                    if (!string.IsNullOrEmpty(cleanWord))
                    {
                        wordFreqs[cleanWord] = wordFreqs.GetValueOrDefault(cleanWord, 0) + 1;
                    }
                }
            }

            // Step 3: Initialize with character vocabulary
            foreach (var word in wordFreqs.Keys)
            {
                foreach (var c in word)
                {
                    vocabulary.AddToken(c.ToString());
                }
            }

            // Step 4: Build subwords using likelihood-based approach
            var subwordCandidates = new Dictionary<string, double>();

            // Generate candidate subwords
            foreach (var (word, freq) in wordFreqs)
            {
                for (int i = 0; i < word.Length; i++)
                {
                    for (int length = 1; length <= Math.Min(word.Length - i, 20); length++)
                    {
                        var subword = word.Substring(i, length);
                        var prefix = i == 0 ? subword : continuingSubwordPrefix + subword;

                        if (!subwordCandidates.ContainsKey(prefix))
                        {
                            subwordCandidates[prefix] = 0;
                        }
                        subwordCandidates[prefix] += freq;
                    }
                }
            }

            // Sort by frequency and add to vocabulary
            var sortedSubwords = subwordCandidates
                .OrderByDescending(kv => kv.Value)
                .Select(kv => kv.Key)
                .ToList();

            foreach (var subword in sortedSubwords)
            {
                if (vocabulary.Size >= vocabSize)
                    break;

                vocabulary.AddToken(subword);
            }

            return new WordPieceTokenizer(vocabulary, specialTokens, continuingSubwordPrefix);
        }

        /// <summary>
        /// Tokenizes text into WordPiece tokens.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var outputTokens = new List<string>();

            // Basic whitespace tokenization
            var words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            var lowercaseWords = words.Select(w => w.ToLowerInvariant());

            foreach (var word in lowercaseWords)
            {
                var wordTokens = TokenizeWord(word);
                outputTokens.AddRange(wordTokens);
            }

            return outputTokens;
        }

        /// <summary>
        /// Tokenizes a single word using WordPiece algorithm.
        /// </summary>
        private List<string> TokenizeWord(string word)
        {
            // Strip punctuation to match training behavior
            var cleanWord = new string(word.Where(c => char.IsLetterOrDigit(c)).ToArray());
            if (string.IsNullOrEmpty(cleanWord))
            {
                return new List<string>();
            }

            if (cleanWord.Length > _maxInputCharsPerWord)
            {
                return new List<string> { SpecialTokens.UnkToken };
            }

            word = cleanWord;

            var tokens = new List<string>();
            int start = 0;

            while (start < word.Length)
            {
                int end = word.Length;
                string? foundSubword = null;

                // Greedy longest-match-first
                while (start < end)
                {
                    var substr = word.Substring(start, end - start);
                    var candidate = start == 0 ? substr : _continuingSubwordPrefix + substr;

                    if (Vocabulary.ContainsToken(candidate))
                    {
                        foundSubword = candidate;
                        break;
                    }

                    end--;
                }

                if (foundSubword == null)
                {
                    // Could not tokenize - use unknown token
                    return new List<string> { SpecialTokens.UnkToken };
                }

                tokens.Add(foundSubword);
                start = end;
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

            var result = new StringBuilder();
            foreach (var token in tokens)
            {
                if (token.StartsWith(_continuingSubwordPrefix))
                {
                    result.Append(token.Substring(_continuingSubwordPrefix.Length));
                }
                else
                {
                    if (result.Length > 0)
                        result.Append(' ');
                    result.Append(token);
                }
            }

            return result.ToString();
        }
    }
}
