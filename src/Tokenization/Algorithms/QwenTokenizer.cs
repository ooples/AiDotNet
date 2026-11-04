using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// Qwen2's text tokenizer (Hugging Face <c>Qwen2TokenizerFast</c> over Qwen/Qwen2.5's <c>tokenizer.json</c>): added
    /// tokens split out of the raw text, NFC, Qwen's pre-tokenizer pattern, and byte-level BPE over the released merges.
    /// </summary>
    /// <remarks>
    /// <para>The vocabulary and merges of Qwen/Qwen2.5-0.5B (Apache-2.0) ship with the library. Its 22 added tokens follow
    /// the 151,643 ranks; <see cref="CosyVoice2"/> adds CosyVoice 2's special tokens after them, as
    /// <c>CosyVoice2Tokenizer</c> does.</para>
    /// <para>The pattern <c>(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+</c>
    /// is matched by a scanner over code points (a .NET regex would read each half of a surrogate pair as a symbol).
    /// Merges apply lowest rank first, the leftmost on ties, one occurrence at a time, as the tokenizers library's
    /// <c>Word::merge_all</c> does.</para>
    /// </remarks>
    internal sealed class QwenTokenizer
    {
        private const string VocabularyResource = "AiDotNet.Tokenization.qwen2-vocab.json.gz";
        private const string MergesResource = "AiDotNet.Tokenization.qwen2-merges.txt.gz";

        /// <summary>Qwen2.5's added tokens, in id order from 151,643.</summary>
        public static readonly IReadOnlyList<string> QwenAddedTokens = new[]
        {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
            "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
        };

        /// <summary>The special tokens <c>CosyVoice2Tokenizer</c> adds (those Qwen already has keep their ids).</summary>
        public static readonly IReadOnlyList<string> CosyVoice2Tokens = new[]
        {
            "<|im_start|>", "<|im_end|>", "<|endofprompt|>", "[breath]", "<strong>", "</strong>", "[noise]", "[laughter]",
            "[cough]", "[clucking]", "[accent]", "[quick_breath]", "<laughter>", "</laughter>", "[hissing]", "[sigh]",
            "[vocalized-noise]", "[lipsmack]", "[mn]",
        };

        /// <summary>The extra system, CMU phoneme and pinyin tokens added by the released CosyVoice 3 tokenizer.</summary>
        public static readonly IReadOnlyList<string> CosyVoice3Tokens = CosyVoice2Tokens.Concat(new[]
        {
            "<|endofsystem|>",
            "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
            "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
            "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
            "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
            "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
            "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
            "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
            "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
            "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
            "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
            "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
            "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
            "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
            "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
            "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
            "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
            "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
            "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
            "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
            "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]",
        }).ToArray();

        private static readonly Lazy<(Dictionary<string, int> Vocabulary, Dictionary<string, int> Merges)> Released = new(Load);
        private static readonly Lazy<QwenTokenizer> CosyVoice2Instance = new(() => new QwenTokenizer(
            QwenAddedTokens.Take(3), CosyVoice2Tokens, "AiDotNet.Tokenization.cosyvoice3-merges.txt.gz"));
        private static readonly Lazy<QwenTokenizer> CosyVoice3Instance = new(() => new QwenTokenizer(
            QwenAddedTokens.Take(3), CosyVoice3Tokens, "AiDotNet.Tokenization.cosyvoice3-merges.txt.gz"));
        private static readonly Lazy<QwenTokenizer> SparkInstance = new(() =>
        {
            using var stream = typeof(QwenTokenizer).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.spark_added_tokens.json.gz")
                ?? throw new InvalidDataException("Missing released Spark-TTS added-token vocabulary.");
            using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
            using var reader = new StreamReader(gzip);
            return new QwenTokenizer(JsonConvert.DeserializeObject<string[]>(reader.ReadToEnd())
                ?? throw new InvalidDataException("Invalid Spark-TTS added-token vocabulary."));
        });

        private readonly Dictionary<string, int> _vocabulary;
        private readonly Dictionary<string, int> _merges;
        private readonly Dictionary<string, int> _added = new(StringComparer.Ordinal);
        private readonly string[] _tokens;
        private readonly Dictionary<byte, char> _byteToChar = BpeTokenizer.BuildByteToUnicode();
        private readonly Dictionary<char, byte> _charToByte;
        private readonly ConcurrentDictionary<string, int[]> _cache = new(StringComparer.Ordinal);

        /// <summary>The tokenizer with Qwen's added tokens and then <paramref name="extraTokens"/> (new ones only).</summary>
        public QwenTokenizer(IEnumerable<string> extraTokens) : this(QwenAddedTokens, extraTokens)
        {
        }

        /// <summary>Qwen2's tokenizer (Qwen2-7B-Instruct's three added tokens: end of text, start and end of a message)
        /// and then <paramref name="extraTokens"/> (new ones only), as a Qwen2-based model's
        /// <c>add_special_tokens</c> extends it.</summary>
        public static QwenTokenizer Qwen2(IEnumerable<string> extraTokens) => new(QwenAddedTokens.Take(3), extraTokens);

        private QwenTokenizer(IEnumerable<string> addedTokens, IEnumerable<string>? extraTokens, string? mergesResource = null)
        {
            var released = Released.Value;
            _vocabulary = released.Vocabulary;
            _merges = mergesResource is null ? released.Merges : LoadMerges(mergesResource);
            int next = _vocabulary.Count;
            foreach (var token in addedTokens.Concat(extraTokens ?? Array.Empty<string>()))
                if (!_added.ContainsKey(token)) _added[token] = next++;
            _tokens = new string[next];
            foreach (var pair in _vocabulary) _tokens[pair.Value] = pair.Key;
            foreach (var pair in _added) _tokens[pair.Value] = pair.Key;
            _charToByte = _byteToChar.ToDictionary(p => p.Value, p => p.Key);
        }

        /// <summary>CosyVoice 2's released BlankEN text tokenizer (151,663 ids; its merge file is shared with CosyVoice 3).</summary>
        public static QwenTokenizer CosyVoice2 => CosyVoice2Instance.Value;

        /// <summary>CosyVoice 3's Qwen2 text tokenizer, including the release's phoneme vocabulary.</summary>
        public static QwenTokenizer CosyVoice3 => CosyVoice3Instance.Value;

        /// <summary>Spark-TTS's Qwen2.5 tokenizer, including all 13,493 added speech and attribute tokens.</summary>
        public static QwenTokenizer Spark => SparkInstance.Value;

        /// <summary>The number of ids: the vocabulary and the added tokens.</summary>
        public int VocabularySize => _tokens.Length;

        /// <summary>The id of an added token.</summary>
        public int AddedTokenId(string token) =>
            _added.TryGetValue(token, out int id) ? id : throw new KeyNotFoundException($"'{token}' is not an added token.");

        /// <summary>The ids of <paramref name="text"/> (<c>tokenizer([text])["input_ids"][0]</c>).</summary>
        public int[] Encode(string text)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            var ids = new List<int>();
            int start = 0, position = 0;
            while (position < text.Length)
            {
                // The longest added token starting here (leftmost-longest, as the tokenizers library matches them).
                string? found = null;
                foreach (var token in _added.Keys)
                    if (string.CompareOrdinal(text, position, token, 0, token.Length) == 0 && (found is null || token.Length > found.Length))
                        found = token;
                if (found is null)
                {
                    position++;
                    continue;
                }
                EncodeOrdinary(text.Substring(start, position - start), ids);
                ids.Add(_added[found]);
                position += found.Length;
                start = position;
            }
            EncodeOrdinary(text.Substring(start), ids);
            return ids.ToArray();
        }

        /// <summary>The text of ids, added tokens included unless <paramref name="skipAdded"/>.</summary>
        public string Decode(IEnumerable<int> ids, bool skipAdded = true)
        {
            if (ids is null) throw new ArgumentNullException(nameof(ids));
            var bytes = new List<byte>();
            foreach (int id in ids)
            {
                if (id < 0 || id >= _tokens.Length) throw new ArgumentOutOfRangeException(nameof(ids), id, $"Token ids run 0 … {_tokens.Length - 1}.");
                string token = _tokens[id];
                if (_added.ContainsKey(token))
                {
                    if (!skipAdded) bytes.AddRange(Encoding.UTF8.GetBytes(token));
                    continue;
                }
                foreach (char c in token) bytes.Add(_charToByte[c]);
            }
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        private void EncodeOrdinary(string segment, List<int> ids)
        {
            if (segment.Length == 0) return;
            string text = segment.Normalize(NormalizationForm.FormC);
            int position = 0;
            while (position < text.Length)
            {
                int length = PreTokenLength(text, position);
                ids.AddRange(_cache.GetOrAdd(text.Substring(position, length), Bpe));
                position += length;
            }
        }

        // Byte-level BPE of one pre-token: its UTF-8 bytes as the GPT-2 alphabet, then the merges, lowest rank first.
        private int[] Bpe(string piece)
        {
            var symbols = Encoding.UTF8.GetBytes(piece).Select(b => _byteToChar[b].ToString()).ToList();
            while (symbols.Count > 1)
            {
                int best = -1, bestRank = int.MaxValue;
                for (int i = 0; i + 1 < symbols.Count; i++)
                    if (_merges.TryGetValue(symbols[i] + "\u0001" + symbols[i + 1], out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        best = i;
                    }
                if (best < 0) break;
                symbols[best] += symbols[best + 1];
                symbols.RemoveAt(best + 1);
            }
            return symbols.Select(s => _vocabulary.TryGetValue(s, out int id) ? id
                : throw new InvalidDataException($"The merges made '{s}', which the vocabulary lacks.")).ToArray();
        }

        // ---------------------------------------------------------------- pre-tokenizer

        private enum Kind { Letter, Number, Space, Other }

        private static Kind KindAt(string text, int index, out int width)
        {
            width = char.IsHighSurrogate(text[index]) && index + 1 < text.Length && char.IsLowSurrogate(text[index + 1]) ? 2 : 1;
            if (width == 1 && char.IsWhiteSpace(text[index])) return Kind.Space;
            switch (CharUnicodeInfo.GetUnicodeCategory(text, index))
            {
                case UnicodeCategory.UppercaseLetter:
                case UnicodeCategory.LowercaseLetter:
                case UnicodeCategory.TitlecaseLetter:
                case UnicodeCategory.ModifierLetter:
                case UnicodeCategory.OtherLetter:
                    return Kind.Letter;
                case UnicodeCategory.DecimalDigitNumber:
                case UnicodeCategory.LetterNumber:
                case UnicodeCategory.OtherNumber:
                    return Kind.Number;
                default:
                    return Kind.Other;
            }
        }

        private static int RunEnd(string text, int index, Kind kind)
        {
            while (index < text.Length && KindAt(text, index, out int width) == kind) index += width;
            return index;
        }

        private static bool IsNewline(char c) => c == '\r' || c == '\n';

        // (?i:...) folds ASCII letters, and ſ (U+017F) to s.
        private static bool Is(char c, char lower) => char.ToLowerInvariant(c) == lower || (lower == 's' && c == 'ſ');

        /// <summary>The length of the pattern's match at <paramref name="position"/>, the alternatives tried in order.</summary>
        private static int PreTokenLength(string text, int position)
        {
            int end = text.Length;
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (text[position] == '\'' && position + 1 < end)
            {
                char a = text[position + 1];
                if (Is(a, 's') || Is(a, 't')) return 2;
                if (position + 2 < end)
                {
                    char b = text[position + 2];
                    if ((Is(a, 'r') && Is(b, 'e')) || (Is(a, 'v') && Is(b, 'e'))) return 3;
                }
                if (Is(a, 'm')) return 2;
                if (position + 2 < end && Is(a, 'l') && Is(text[position + 2], 'l')) return 3;
                if (Is(a, 'd')) return 2;
            }
            var kind = KindAt(text, position, out int width);
            // [^\r\n\p{L}\p{N}]?\p{L}+
            if (kind != Kind.Letter && kind != Kind.Number && !IsNewline(text[position]) && position + width < end
                && KindAt(text, position + width, out _) == Kind.Letter)
                return RunEnd(text, position + width, Kind.Letter) - position;
            if (kind == Kind.Letter) return RunEnd(text, position, Kind.Letter) - position;
            // \p{N}
            if (kind == Kind.Number) return width;
            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            int from = text[position] == ' ' && position + 1 < end && KindAt(text, position + 1, out _) == Kind.Other ? position + 1 : position;
            if (KindAt(text, from, out _) == Kind.Other)
            {
                int stop = RunEnd(text, from, Kind.Other);
                while (stop < end && IsNewline(text[stop])) stop++;
                return stop - position;
            }
            // \s*[\r\n]+ : the whitespace run up to its last newline.
            int spaceEnd = RunEnd(text, position, Kind.Space);
            for (int i = spaceEnd - 1; i >= position; i--)
                if (IsNewline(text[i])) return i + 1 - position;
            // \s+(?!\S) | \s+
            int run = spaceEnd - position;
            return spaceEnd == end || run == 1 ? run : run - 1;
        }

        private static string ReadResource(string resource)
        {
            using var stream = typeof(QwenTokenizer).Assembly.GetManifestResourceStream(resource)
                ?? throw new InvalidDataException($"The embedded resource {resource} is missing.");
            using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
            using var reader = new StreamReader(gzip, Encoding.UTF8);
            return reader.ReadToEnd();
        }

        private static (Dictionary<string, int> Vocabulary, Dictionary<string, int> Merges) Load()
        {
            var vocabulary = JsonConvert.DeserializeObject<Dictionary<string, int>>(ReadResource(VocabularyResource))
                ?? throw new InvalidDataException("The Qwen2 vocabulary is empty.");
            return (new Dictionary<string, int>(vocabulary, StringComparer.Ordinal), LoadMerges(MergesResource));
        }

        private static Dictionary<string, int> LoadMerges(string resource)
        {
            var merges = new Dictionary<string, int>(StringComparer.Ordinal);
            int rank = 0;
            foreach (var line in ReadResource(resource).Split('\n'))
            {
                var trimmed = line.TrimEnd('\r');
                if (trimmed.Length == 0 || trimmed.StartsWith("#version", StringComparison.Ordinal)) continue;
                int space = trimmed.IndexOf(' ');
                if (space <= 0 || space == trimmed.Length - 1) throw new InvalidDataException("Invalid Qwen merge pair.");
                merges[trimmed.Substring(0, space) + "\u0001" + trimmed.Substring(space + 1)] = rank++;
            }
            return merges;
        }
    }
}
