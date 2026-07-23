using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Validation;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// SentencePiece tokenizer implementation using Unigram language model.
    /// Used for multilingual models and language-agnostic tokenization.
    /// </summary>
    public class SentencePieceTokenizer : TokenizerBase
    {
        private readonly Dictionary<string, double> _pieceScores;
        private readonly bool _treatWhitespaceAsSpecialToken;
        private const string WhitespaceSymbol = "▁";

        /// <summary>
        /// Creates a new SentencePiece tokenizer.
        /// </summary>
        /// <param name="vocabulary">The vocabulary.</param>
        /// <param name="pieceScores">The scores for each piece (used for unigram segmentation).</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="treatWhitespaceAsSpecialToken">Whether to treat whitespace as a special token.</param>
        public SentencePieceTokenizer(
            IVocabulary vocabulary,
            Dictionary<string, double> pieceScores,
            SpecialTokens? specialTokens = null,
            bool treatWhitespaceAsSpecialToken = true)
            : base(vocabulary, specialTokens ?? SpecialTokens.T5())
        {
            Guard.NotNull(pieceScores);
            _pieceScores = pieceScores;
            _treatWhitespaceAsSpecialToken = treatWhitespaceAsSpecialToken;
        }

        /// <summary>
        /// Trains a SentencePiece tokenizer using Unigram language model.
        /// </summary>
        /// <param name="corpus">The training corpus.</param>
        /// <param name="vocabSize">The desired vocabulary size.</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="characterCoverage">Character coverage (default: 0.9995).</param>
        /// <returns>A trained SentencePiece tokenizer.</returns>
        public static SentencePieceTokenizer Train(
            IEnumerable<string> corpus,
            int vocabSize,
            SpecialTokens? specialTokens = null,
            double characterCoverage = 0.9995)
        {
            specialTokens ??= SpecialTokens.T5();

            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);

            // Add special tokens first
            foreach (var token in specialTokens.GetAllSpecialTokens())
            {
                vocabulary.AddToken(token);
            }

            // Step 1: Character frequency analysis
            var charFreqs = new Dictionary<char, int>();
            foreach (var text in corpus)
            {
                foreach (var c in text)
                {
                    charFreqs[c] = charFreqs.GetValueOrDefault(c, 0) + 1;
                }
            }

            // Step 2: Select characters based on coverage
            var totalChars = charFreqs.Values.Sum();
            var sortedChars = charFreqs.OrderByDescending(kv => kv.Value).ToList();
            var selectedChars = new HashSet<char>();
            int charCount = 0;

            foreach (var (c, freq) in sortedChars)
            {
                selectedChars.Add(c);
                charCount += freq;
                if ((double)charCount / totalChars >= characterCoverage)
                    break;
            }

            // Step 3: Initialize seed vocabulary with characters
            var pieceScores = new Dictionary<string, double>();

            foreach (var token in selectedChars.Select(c => c.ToString()))
            {
                vocabulary.AddToken(token);
                pieceScores[token] = 0.0; // Initial score
            }

            // Step 4: Generate subword candidates
            var subwordCandidates = new Dictionary<string, int>();

            foreach (var processedText in corpus.Select(t => t.Replace(" ", WhitespaceSymbol)))
            {
                // Generate subwords
                for (int i = 0; i < processedText.Length; i++)
                {
                    for (int length = 2; length <= Math.Min(processedText.Length - i, 20); length++)
                    {
                        var subword = processedText.Substring(i, length);
                        if (subword.All(c => selectedChars.Contains(c) || c == WhitespaceSymbol[0]))
                        {
                            subwordCandidates[subword] = subwordCandidates.GetValueOrDefault(subword, 0) + 1;
                        }
                    }
                }
            }

            // Step 5: Score and select top subwords
            var scoredSubwords = subwordCandidates
                .Select(kv => (Subword: kv.Key, Score: Math.Log(kv.Value)))
                .OrderByDescending(s => s.Score)
                .ToList();

            foreach (var (subword, score) in scoredSubwords)
            {
                if (vocabulary.Size >= vocabSize)
                    break;

                vocabulary.AddToken(subword);
                pieceScores[subword] = score;
            }

            return new SentencePieceTokenizer(vocabulary, pieceScores, specialTokens);
        }

        /// <summary>
        /// Tokenizes text into SentencePiece tokens.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            // Replace spaces with whitespace symbol
            var processedText = text.Replace(" ", WhitespaceSymbol);

            if (_treatWhitespaceAsSpecialToken)
            {
                // Split on whitespace symbol and tokenize each segment separately
                // This ensures whitespace symbols are kept as separate tokens
                var segments = processedText.Split(new[] { WhitespaceSymbol }, StringSplitOptions.None);
                var tokens = new List<string>();

                for (int i = 0; i < segments.Length; i++)
                {
                    // Add whitespace symbol token before each segment except the first
                    if (i > 0)
                    {
                        tokens.Add(WhitespaceSymbol);
                    }

                    // Tokenize the segment if it's not empty
                    if (!string.IsNullOrEmpty(segments[i]))
                    {
                        tokens.AddRange(ViterbiSegmentation(segments[i]));
                    }
                }

                return tokens;
            }
            else
            {
                // Use Viterbi algorithm on the entire text, allowing whitespace to merge
                return ViterbiSegmentation(processedText);
            }
        }

        /// <summary>
        /// Performs Viterbi segmentation to find the best tokenization.
        /// </summary>
        private List<string> ViterbiSegmentation(string text)
        {
            if (text.Length == 0)
                return new List<string>();

            int n = text.Length;
            var scores = new double[n + 1];
            var backtrack = new int[n + 1];

            // Initialize
            for (int i = 0; i <= n; i++)
            {
                scores[i] = double.NegativeInfinity;
                backtrack[i] = -1;
            }
            scores[0] = 0;

            // Forward pass
            for (int i = 0; i < n; i++)
            {
                if (double.IsNegativeInfinity(scores[i]))
                    continue;

                for (int j = i + 1; j <= n; j++)
                {
                    var piece = text.Substring(i, j - i);

                    if (!Vocabulary.ContainsToken(piece))
                        continue;

                    var pieceScore = _pieceScores.GetValueOrDefault(piece, -10.0);
                    var newScore = scores[i] + pieceScore;

                    if (newScore > scores[j])
                    {
                        scores[j] = newScore;
                        backtrack[j] = i;
                    }
                }
            }

            // Backward pass to reconstruct tokens
            var tokens = new List<string>();
            int pos = n;

            while (pos > 0)
            {
                if (backtrack[pos] == -1)
                {
                    // Fallback: emit single character as unknown and continue
                    // This ensures we don't lose characters when no valid path exists
                    tokens.Insert(0, SpecialTokens.UnkToken);
                    pos--;
                    continue;
                }

                var start = backtrack[pos];
                var piece = text.Substring(start, pos - start);
                tokens.Insert(0, piece);
                pos = start;
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

            var result = string.Join("", tokens);

            // Replace whitespace symbol with space
            result = result.Replace(WhitespaceSymbol, " ");

            return result.Trim();
        }
    }
}
