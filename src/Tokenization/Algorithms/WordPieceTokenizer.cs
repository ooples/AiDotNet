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
        private readonly bool _lowercase;
        private readonly bool _stripAccents;

        private const string BertBaseUncasedResource = "AiDotNet.Tokenization.bert-base-uncased-vocab.txt.gz";
        private const string BertBaseChineseResource = "AiDotNet.Tokenization.bert-base-chinese-vocab.txt.gz";
        private const string BertBaseMultilingualUncasedResource = "AiDotNet.Tokenization.bert-base-multilingual-uncased-vocab.txt.gz";

        /// <summary>
        /// BERT-base's uncased WordPiece tokenizer (Devlin et al. 2019): the released 30,522-token vocabulary
        /// (google-research/bert, Apache-2.0), shipped with the library, with lowercasing and accent stripping.
        /// </summary>
        public static WordPieceTokenizer BertBaseUncased() =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(BertBaseUncasedResource, null), "[UNK]"), SpecialTokens.Bert());

        /// <summary>
        /// BERT-base Chinese's WordPiece tokenizer: the released 21,128-token vocabulary (google-research/bert,
        /// Apache-2.0), shipped with the library, with each CJK character its own word, lowercasing and accent
        /// stripping. <paramref name="additionalTokens"/> take the ids after it, in order, as a Hugging Face tokenizer's
        /// added tokens do (ChatTTS adds its 50 control tokens this way).
        /// </summary>
        public static WordPieceTokenizer BertBaseChinese(IEnumerable<string>? additionalTokens = null) =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(BertBaseChineseResource, additionalTokens), "[UNK]"), SpecialTokens.Bert());

        /// <summary>
        /// Multilingual BERT-base's uncased WordPiece tokenizer: the released 105,879-token vocabulary
        /// (google-research/bert, Apache-2.0), shipped with the library, with each CJK character its own word,
        /// lowercasing and accent stripping. MeloTTS's mixed Chinese-English front end tokenizes its English with it.
        /// </summary>
        public static WordPieceTokenizer BertBaseMultilingualUncased() =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(BertBaseMultilingualUncasedResource, null), "[UNK]"), SpecialTokens.Bert());

        private const string BertBaseSpanishWwmUncasedResource = "AiDotNet.Tokenization.bert-base-spanish-wwm-uncased-vocab.txt.gz";
        private const string BertBaseFrenchEuropeanaCasedResource = "AiDotNet.Tokenization.bert-base-french-europeana-cased-vocab.txt.gz";

        /// <summary>
        /// BETO's WordPiece tokenizer (dccuchile/bert-base-spanish-wwm-uncased, CC BY 4.0): its 31,002-token vocabulary,
        /// shipped with the library, lowercased with accents kept (its tokenizer_config sets strip_accents false).
        /// MeloTTS's Spanish front end tokenizes with it.
        /// </summary>
        public static WordPieceTokenizer BertBaseSpanishWwmUncased() =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(BertBaseSpanishWwmUncasedResource, null), "[UNK]"), SpecialTokens.Bert(),
                lowercase: true, stripAccents: false);

        /// <summary>
        /// dbmdz's cased French Europeana BERT WordPiece tokenizer (dbmdz/bert-base-french-europeana-cased, MIT): its
        /// 32,000-token vocabulary, shipped with the library, neither lowercased nor accent-stripped. MeloTTS's French
        /// front end tokenizes with it.
        /// </summary>
        public static WordPieceTokenizer BertBaseFrenchEuropeanaCased() =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(BertBaseFrenchEuropeanaCasedResource, null), "[UNK]"), SpecialTokens.Bert(),
                lowercase: false);

        private const string BertBaseJapaneseV3Resource = "AiDotNet.Tokenization.bert-base-japanese-v3-vocab.txt.gz";

        /// <summary>
        /// The WordPiece half of tohoku-nlp/bert-base-japanese-v3's tokenizer (Apache-2.0): its 32,768-token vocabulary,
        /// shipped with the library, case kept. The words it splits come from MeCab with unidic-lite
        /// (the library's MeCab port); pass them to <see cref="TokenizeWords"/>.
        /// </summary>
        public static WordPieceTokenizer BertBaseJapaneseV3() =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(BertBaseJapaneseV3Resource, null), "[UNK]"), SpecialTokens.Bert(),
                lowercase: false);

        private const string SimBertChineseResource = "AiDotNet.Tokenization.simbert-base-chinese-vocab.txt.gz";

        /// <summary>
        /// SimBERT's WordPiece tokenizer (WangZeJun/simbert-base-chinese, which EmotiVoice's style encoder reads): its
        /// 13,685-token Chinese vocabulary, a subset of BERT-base Chinese's, with each CJK character its own word and
        /// lowercasing.
        /// </summary>
        public static WordPieceTokenizer SimBertChinese() =>
            new WordPieceTokenizer(new Vocabulary.Vocabulary(LoadVocabulary(SimBertChineseResource, null), "[UNK]"), SpecialTokens.Bert());

        private static Dictionary<string, int> LoadVocabulary(string resource, IEnumerable<string>? additionalTokens)
        {
            var assembly = typeof(WordPieceTokenizer).Assembly;
            using var stream = assembly.GetManifestResourceStream(resource)
                ?? throw new System.IO.InvalidDataException($"The embedded resource {resource} is missing.");
            using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
            using var reader = new System.IO.StreamReader(gzip, Encoding.UTF8);
            var ids = new Dictionary<string, int>(StringComparer.Ordinal);
            string? line;
            while ((line = reader.ReadLine()) is not null)
                if (!ids.ContainsKey(line)) ids[line] = ids.Count;
            if (additionalTokens is not null)
                foreach (var token in additionalTokens)
                {
                    if (ids.ContainsKey(token)) throw new ArgumentException($"'{token}' is already in the vocabulary.", nameof(additionalTokens));
                    ids[token] = ids.Count;
                }
            return ids;
        }

        /// <summary>
        /// Creates a new WordPiece tokenizer.
        /// </summary>
        /// <param name="vocabulary">The vocabulary.</param>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="continuingSubwordPrefix">The prefix for continuing subwords (default: "##").</param>
        /// <param name="maxInputCharsPerWord">Maximum characters per word (default: 100).</param>
        /// <param name="lowercase">Lowercase the text first (uncased models).</param>
        /// <param name="stripAccents">Strip accents (Hugging Face's strip_accents); null follows
        /// <paramref name="lowercase"/>, as BERT's tokenizer does.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if maxInputCharsPerWord is less than 1.</exception>
        public WordPieceTokenizer(
            IVocabulary vocabulary,
            SpecialTokens? specialTokens = null,
            string continuingSubwordPrefix = "##",
            int maxInputCharsPerWord = 100,
            bool lowercase = true,
            bool? stripAccents = null)
            : base(vocabulary, specialTokens ?? SpecialTokens.Bert())
        {
            if (maxInputCharsPerWord < 1)
                throw new ArgumentOutOfRangeException(nameof(maxInputCharsPerWord), "Max input chars per word must be at least 1.");

            _continuingSubwordPrefix = continuingSubwordPrefix;
            _maxInputCharsPerWord = maxInputCharsPerWord;
            _lowercase = lowercase;
            _stripAccents = stripAccents ?? lowercase;
        }

        /// <summary>
        /// BERT's basic tokenization (the reference BasicTokenizer): control characters removed and whitespace
        /// normalized, each CJK character made its own word, words split on whitespace, lowercased with accents
        /// stripped (uncased models), and every punctuation character split off as its own word.
        /// </summary>
        internal static List<string> BasicTokenize(string text, bool lowercase, bool? stripAccents = null)
        {
            var cleaned = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                if (c == 0 || c == 0xFFFD || IsControl(c)) continue;
                if (IsWhitespace(c)) cleaned.Append(' ');
                else if (IsCjk(c)) cleaned.Append(' ').Append(c).Append(' ');
                else cleaned.Append(c);
            }
            var words = new List<string>();
            foreach (var token in cleaned.ToString().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                string word = token;
                if (lowercase) word = word.ToLowerInvariant();
                if (stripAccents ?? lowercase)
                {
                    var decomposed = word.Normalize(NormalizationForm.FormD);
                    word = new string(decomposed.Where(ch => System.Globalization.CharUnicodeInfo.GetUnicodeCategory(ch)
                        != System.Globalization.UnicodeCategory.NonSpacingMark).ToArray());
                }
                var current = new StringBuilder();
                foreach (char ch in word)
                {
                    if (IsPunctuation(ch))
                    {
                        if (current.Length > 0) { words.Add(current.ToString()); current.Clear(); }
                        words.Add(ch.ToString());
                    }
                    else current.Append(ch);
                }
                if (current.Length > 0) words.Add(current.ToString());
            }
            return words;
        }

        private static bool IsWhitespace(char c)
            => c == ' ' || c == '\t' || c == '\n' || c == '\r'
               || System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c) == System.Globalization.UnicodeCategory.SpaceSeparator;

        private static bool IsControl(char c)
        {
            if (c == '\t' || c == '\n' || c == '\r') return false;
            var category = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c);
            return category == System.Globalization.UnicodeCategory.Control || category == System.Globalization.UnicodeCategory.Format;
        }

        // ASCII non-letters/digits count as punctuation even where Unicode disagrees (^, $, `), as in the reference.
        private static bool IsPunctuation(char c)
        {
            if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true;
            switch (System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c))
            {
                case System.Globalization.UnicodeCategory.ConnectorPunctuation:
                case System.Globalization.UnicodeCategory.DashPunctuation:
                case System.Globalization.UnicodeCategory.OpenPunctuation:
                case System.Globalization.UnicodeCategory.ClosePunctuation:
                case System.Globalization.UnicodeCategory.InitialQuotePunctuation:
                case System.Globalization.UnicodeCategory.FinalQuotePunctuation:
                case System.Globalization.UnicodeCategory.OtherPunctuation:
                    return true;
                default:
                    return false;
            }
        }

        private static bool IsCjk(char c)
            => (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0xF900 && c <= 0xFAFF);

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
                // The same pre-tokenization as encoding: punctuation is its own word, and so its own token.
                foreach (var word in BasicTokenize(text, lowercase: true))
                {
                    wordFreqs[word] = wordFreqs.GetValueOrDefault(word, 0) + 1;
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
            foreach (var word in BasicTokenize(text, _lowercase, _stripAccents))
                outputTokens.AddRange(TokenizeWord(word));
            return outputTokens;
        }

        /// <summary>
        /// WordPiece over words another tokenizer produced, as Hugging Face's <c>WordpieceTokenizer.tokenize</c> runs
        /// on each (BertJapaneseTokenizer feeds it MeCab's words): each split on whitespace, then matched greedily.
        /// </summary>
        public List<string> TokenizeWords(IEnumerable<string> words)
        {
            var outputTokens = new List<string>();
            foreach (var word in words)
                foreach (var piece in word.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries))
                    outputTokens.AddRange(TokenizeWord(piece));
            return outputTokens;
        }

        /// <summary>
        /// Tokenizes a single word using WordPiece algorithm.
        /// </summary>
        private List<string> TokenizeWord(string word)
        {
            if (string.IsNullOrEmpty(word))
            {
                return new List<string>();
            }

            if (word.Length > _maxInputCharsPerWord)
            {
                return new List<string> { SpecialTokens.UnkToken };
            }

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
