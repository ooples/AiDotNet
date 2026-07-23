using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// Character-level tokenizer that splits text into individual characters.
    /// Useful for character-based language models and some RNN architectures.
    /// </summary>
    public class CharacterTokenizer : TokenizerBase
    {
        private readonly bool _lowercase;
        private readonly bool _includeWhitespace;

        /// <summary>
        /// Creates a new character tokenizer.
        /// </summary>
        public CharacterTokenizer(
            IVocabulary vocabulary,
            SpecialTokens specialTokens,
            bool lowercase = false,
            bool includeWhitespace = true)
            : base(vocabulary, specialTokens)
        {
            _lowercase = lowercase;
            _includeWhitespace = includeWhitespace;
        }

        /// <summary>
        /// Tokenizes text into individual characters.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var processedText = _lowercase ? text.ToLowerInvariant() : text;
            var characters = new List<string>();

            foreach (char c in processedText)
            {
                if (!_includeWhitespace && char.IsWhiteSpace(c))
                {
                    if (characters.Count == 0 || characters[characters.Count - 1] != " ")
                        characters.Add(" ");
                }
                else
                {
                    var charStr = c.ToString();
                    characters.Add(Vocabulary.ContainsToken(charStr) ? charStr : SpecialTokens.UnkToken);
                }
            }

            return characters;
        }

        /// <summary>
        /// Cleans up tokens and converts them back to text.
        /// </summary>
        protected override string CleanupTokens(List<string> tokens)
        {
            return string.Concat(tokens);
        }

        /// <summary>
        /// Trains a character tokenizer from a corpus.
        /// </summary>
        /// <exception cref="ArgumentNullException">Thrown if corpus is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if minFrequency is less than 1.</exception>
        public static CharacterTokenizer Train(
            IEnumerable<string> corpus,
            SpecialTokens? specialTokens = null,
            bool lowercase = false,
            int minFrequency = 1)
        {
            if (corpus == null)
                throw new ArgumentNullException(nameof(corpus));
            if (minFrequency < 1)
                throw new ArgumentOutOfRangeException(nameof(minFrequency), "Min frequency must be at least 1.");

            specialTokens ??= SpecialTokens.Default();
            var charFrequencies = new Dictionary<char, int>();

            var processedTexts = corpus.Select(text => lowercase ? text.ToLowerInvariant() : text);
            foreach (var processedText in processedTexts)
            {
                foreach (char c in processedText)
                {
                    if (!charFrequencies.ContainsKey(c))
                        charFrequencies[c] = 0;
                    charFrequencies[c]++;
                }
            }

            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());

            var validChars = charFrequencies
                .Where(kvp => kvp.Value >= minFrequency)
                .OrderByDescending(kvp => kvp.Value)
                .Select(kvp => kvp.Key.ToString());

            vocabulary.AddTokens(validChars);

            return new CharacterTokenizer(vocabulary, specialTokens, lowercase);
        }

        /// <summary>
        /// Creates a character tokenizer with ASCII printable characters.
        /// </summary>
        public static CharacterTokenizer CreateAscii(SpecialTokens? specialTokens = null, bool lowercase = false)
        {
            specialTokens ??= SpecialTokens.Default();
            var asciiChars = Enumerable.Range(32, 95).Select(i => (char)i);

            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());
            vocabulary.AddTokens(asciiChars.Select(c => c.ToString()));

            return new CharacterTokenizer(vocabulary, specialTokens, lowercase);
        }
    }
}
