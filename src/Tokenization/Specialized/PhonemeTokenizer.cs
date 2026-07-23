using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Validation;

namespace AiDotNet.Tokenization.Specialized
{
    /// <summary>
    /// Phoneme-based tokenizer for speech synthesis (TTS) applications.
    /// </summary>
    public class PhonemeTokenizer : TokenizerBase
    {
        private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
        private readonly Dictionary<string, string> _g2pRules;
        private readonly PhonemeSet _phonemeSet;

        /// <summary>
        /// Supported phoneme sets.
        /// </summary>
        public enum PhonemeSet { IPA, ARPAbet, XSAMPA, Custom }

        /// <summary>
        /// Creates a new phoneme tokenizer.
        /// </summary>
        public PhonemeTokenizer(
            IVocabulary vocabulary,
            Dictionary<string, string> g2pRules,
            SpecialTokens specialTokens,
            PhonemeSet phonemeSet = PhonemeSet.IPA)
            : base(vocabulary, specialTokens)
        {
            Guard.NotNull(g2pRules);
            _g2pRules = g2pRules;
            _phonemeSet = phonemeSet;
        }

        /// <summary>
        /// Tokenizes text into phonemes.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var phonemes = new List<string>();
            // Regex matches word characters (\w+) or non-whitespace punctuation ([^\w\s])
            // Whitespace is never matched, so no IsNullOrWhiteSpace check needed
            var words = Regex.Matches(text, @"(\w+|[^\w\s])", RegexOptions.None, RegexTimeout).Cast<Match>().Select(m => m.Value);

            foreach (var word in words)
            {
                var wordPhonemes = ConvertToPhonemes(word.ToLowerInvariant());
                phonemes.AddRange(wordPhonemes);
                phonemes.Add("<space>");
            }

            if (phonemes.Count > 0 && phonemes[phonemes.Count - 1] == "<space>")
                phonemes.RemoveAt(phonemes.Count - 1);

            return phonemes;
        }

        private List<string> ConvertToPhonemes(string word)
        {
            if (_g2pRules.TryGetValue(word, out string? pronunciation) && pronunciation != null)
                return pronunciation.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).ToList();

            return RuleBasedG2P(word);
        }

        private List<string> RuleBasedG2P(string word)
        {
            var phonemes = new List<string>();
            int i = 0;

            while (i < word.Length)
            {
                bool matched = false;
                for (int len = Math.Min(4, word.Length - i); len > 0; len--)
                {
                    var grapheme = word.Substring(i, len);
                    if (_g2pRules.TryGetValue(grapheme, out string? phoneme) && phoneme != null)
                    {
                        phonemes.AddRange(phoneme.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));
                        i += len;
                        matched = true;
                        break;
                    }
                }

                if (!matched)
                {
                    var defaultPhoneme = GetDefaultPhoneme(word[i]);
                    if (!string.IsNullOrEmpty(defaultPhoneme))
                        phonemes.Add(defaultPhoneme);
                    i++;
                }
            }

            return phonemes;
        }

        private string GetDefaultPhoneme(char c)
        {
            var mappings = _phonemeSet == PhonemeSet.ARPAbet
                ? new Dictionary<char, string>
                {
                    {'a', "AE"}, {'b', "B"}, {'c', "K"}, {'d', "D"}, {'e', "EH"}, {'f', "F"},
                    {'g', "G"}, {'h', "HH"}, {'i', "IH"}, {'j', "JH"}, {'k', "K"}, {'l', "L"},
                    {'m', "M"}, {'n', "N"}, {'o', "AA"}, {'p', "P"}, {'q', "K"}, {'r', "R"}, {'s', "S"},
                    {'t', "T"}, {'u', "AH"}, {'v', "V"}, {'w', "W"}, {'x', "K S"}, {'y', "Y"}, {'z', "Z"}
                }
                : new Dictionary<char, string>
                {
                    {'a', "ae"}, {'b', "b"}, {'c', "k"}, {'d', "d"}, {'e', "E"}, {'f', "f"},
                    {'g', "g"}, {'h', "h"}, {'i', "I"}, {'j', "dZ"}, {'k', "k"}, {'l', "l"},
                    {'m', "m"}, {'n', "n"}, {'o', "A"}, {'p', "p"}, {'q', "k"}, {'r', "r"}, {'s', "s"},
                    {'t', "t"}, {'u', "V"}, {'v', "v"}, {'w', "w"}, {'x', "ks"}, {'y', "j"}, {'z', "z"}
                };

            return mappings.TryGetValue(char.ToLowerInvariant(c), out string? phoneme) ? phoneme ?? string.Empty : string.Empty;
        }

        protected override string CleanupTokens(List<string> tokens)
        {
            return string.Join(" ", tokens.Where(t => t != "<space>"));
        }

        /// <summary>
        /// Creates a phoneme tokenizer with ARPAbet phonemes.
        /// </summary>
        public static PhonemeTokenizer CreateARPAbet(SpecialTokens? specialTokens = null)
        {
            specialTokens ??= new SpecialTokens { UnkToken = "<unk>", PadToken = "<pad>" };

            var phonemes = new[] { "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
                "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH" };

            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());
            vocabulary.AddTokens(phonemes);
            vocabulary.AddToken("<space>");

            var g2pRules = new Dictionary<string, string>
            {
                {"th", "TH"}, {"sh", "SH"}, {"ch", "CH"}, {"ng", "NG"}, {"ph", "F"},
                {"ee", "IY"}, {"ea", "IY"}, {"oo", "UW"}, {"ou", "AW"}, {"ai", "EY"}, {"ay", "EY"}
            };

            return new PhonemeTokenizer(vocabulary, g2pRules, specialTokens, PhonemeSet.ARPAbet);
        }
    }
}
