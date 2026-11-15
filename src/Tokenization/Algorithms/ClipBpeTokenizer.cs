using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// CLIP's byte-level BPE tokenizer, as open_clip's <c>SimpleTokenizer</c> (and OpenAI CLIP's) implements it, with its
    /// vocabulary loaded from the merges file <c>bpe_simple_vocab_16e6.txt.gz</c> (gzip or plain text, from a path or a
    /// stream).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The vocabulary is built as <c>SimpleTokenizer.__init__</c> builds it: the 256 byte characters of
    /// <c>bytes_to_unicode()</c>, the same 256 with <c>&lt;/w&gt;</c>, one token per merge (the file's lines after its
    /// header, at most 48,894), then <c>&lt;start_of_text&gt;</c> and <c>&lt;end_of_text&gt;</c>: 49,408 tokens for the
    /// released file, with the special tokens at 49406 and 49407.
    /// </para>
    /// <para>
    /// Text is cleaned as <c>whitespace_clean(basic_clean(text)).lower()</c>: HTML entities unescaped twice
    /// (<c>html.unescape(html.unescape(text))</c>), stripped, every whitespace run collapsed to one space, lower-cased.
    /// It is then split by CLIP's pattern (the special tokens, <c>'s|'t|'re|'ve|'m|'ll|'d</c>, letter runs, single digits,
    /// and runs of other non-space characters), each piece mapped to its UTF-8 bytes' characters and merged by rank.
    /// </para>
    /// <para>
    /// Two documented differences from open_clip's cleaning:
    /// </para>
    /// <list type="bullet">
    /// <item><description><c>ftfy.fix_text</c> is not applied. open_clip runs it first; with ftfy's default settings
    /// it repairs mojibake (text decoded with the wrong codec, such as <c>"â€™"</c> for <c>"’"</c>), uncurls quotes,
    /// expands Latin ligatures, maps full-width characters to their normal width, normalizes line breaks, removes
    /// terminal escapes and control characters, fixes surrogates, and normalizes to Unicode NFC. Text with none of those
    /// (clean, NFC-normalized UTF-8 with straight quotes, no ligatures or full-width forms) tokenizes identically; clean
    /// or normalize other text before tokenizing.</description></item>
    /// <item><description>HTML unescaping is <see cref="WebUtility.HtmlDecode(string)"/>, which knows HTML 4's 253 named
    /// entities and numeric references. Python's <c>html.unescape</c> also knows the rest of HTML 5's named entities
    /// (and a few without the closing semicolon); text using those keeps them unescaped here.</description></item>
    /// </list>
    /// <para>
    /// <see cref="EncodeContext"/> is <c>open_clip.tokenize</c>: <c>[&lt;start_of_text&gt;] + tokens +
    /// [&lt;end_of_text&gt;]</c>, cut to the context length keeping <c>&lt;end_of_text&gt;</c> last, zero-padded.
    /// </para>
    /// </remarks>
    public sealed class ClipBpeTokenizer : TokenizerBase
    {
        /// <summary>The start-of-text token.</summary>
        public const string StartOfTextToken = "<start_of_text>";

        /// <summary>The end-of-text token.</summary>
        public const string EndOfTextToken = "<end_of_text>";

        /// <summary>The number of merges the released vocabulary uses (<c>49152 - 256 - 2</c>).</summary>
        public const int MaxMerges = 49152 - 256 - 2;

        private const string WordEnd = "</w>";
        private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

        // SimpleTokenizer's pattern (the special tokens, contractions, letters, digits, other runs), IGNORECASE.
        private static readonly Regex Pattern = new Regex(
            @"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, RegexTimeout);

        private static readonly Regex Whitespace = new Regex(@"\s+", RegexOptions.CultureInvariant, RegexTimeout);

        private readonly Dictionary<string, int> _encoder;
        private readonly Dictionary<(string, string), int> _ranks;
        private readonly char[] _byteEncoder;
        private readonly Dictionary<char, byte> _byteDecoder;
        private readonly ConcurrentDictionary<string, string> _cache = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);

        private ClipBpeTokenizer(Dictionary<string, int> encoder, Dictionary<(string, string), int> ranks, char[] byteEncoder)
            : base(new global::AiDotNet.Tokenization.Vocabulary.Vocabulary(encoder, EndOfTextToken), new SpecialTokens
            {
                UnkToken = EndOfTextToken,
                PadToken = byteEncoder['!'].ToString(),
                ClsToken = StartOfTextToken,
                SepToken = EndOfTextToken,
                BosToken = StartOfTextToken,
                EosToken = EndOfTextToken,
            })
        {
            _encoder = encoder;
            _ranks = ranks;
            _byteEncoder = byteEncoder;
            _byteDecoder = new Dictionary<char, byte>();
            for (int b = 0; b < 256; b++) _byteDecoder[byteEncoder[b]] = (byte)b;
            StartOfTextId = encoder[StartOfTextToken];
            EndOfTextId = encoder[EndOfTextToken];
        }

        /// <summary>The id of <c>&lt;start_of_text&gt;</c> (49406 for the released vocabulary).</summary>
        public int StartOfTextId { get; }

        /// <summary>The id of <c>&lt;end_of_text&gt;</c> (49407 for the released vocabulary).</summary>
        public int EndOfTextId { get; }

        /// <summary>Loads the merges file (<c>bpe_simple_vocab_16e6.txt.gz</c>, gzip or plain text) from a path.</summary>
        public static ClipBpeTokenizer FromFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("A merges file path is required.", nameof(path));
            using var stream = File.OpenRead(path);
            return FromStream(stream);
        }

        /// <summary>Loads the merges file (gzip or plain text, detected from its first bytes) from a stream.</summary>
        public static ClipBpeTokenizer FromStream(Stream stream)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            var bytes = buffer.ToArray();
            string text;
            if (bytes.Length >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b)
            {
                using var gzip = new GZipStream(new MemoryStream(bytes), CompressionMode.Decompress);
                using var reader = new StreamReader(gzip, new UTF8Encoding(false));
                text = reader.ReadToEnd();
            }
            else
            {
                text = new UTF8Encoding(false).GetString(bytes);
            }
            return FromMerges(text.Split('\n'));
        }

        /// <summary>
        /// Builds the tokenizer from the merges file's lines: the first line is the header and is skipped, as
        /// <c>merges[1:49152-256-2+1]</c> skips it; each further line is one merge, <c>"first second"</c>. Blank lines
        /// (the file's trailing newline) are ignored.
        /// </summary>
        public static ClipBpeTokenizer FromMerges(IEnumerable<string> lines)
        {
            if (lines is null) throw new ArgumentNullException(nameof(lines));
            var merges = new List<(string, string)>();
            foreach (var raw in lines.Skip(1))
            {
                if (merges.Count == MaxMerges) break;
                var line = raw.TrimEnd('\r');
                if (line.Length == 0) continue;
                var parts = line.Split(' ');
                if (parts.Length != 2 || parts[0].Length == 0 || parts[1].Length == 0)
                    throw new InvalidDataException($"A CLIP merge is two space-separated symbols; got '{line}'.");
                merges.Add((parts[0], parts[1]));
            }

            var byteEncoder = BytesToUnicode();
            var vocab = new List<string>(512 + merges.Count + 2);
            var order = ByteOrder();
            foreach (int b in order) vocab.Add(byteEncoder[b].ToString());
            foreach (int b in order) vocab.Add(byteEncoder[b] + WordEnd);
            foreach (var (first, second) in merges) vocab.Add(first + second);
            vocab.Add(StartOfTextToken);
            vocab.Add(EndOfTextToken);

            var encoder = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < vocab.Count; i++) encoder[vocab[i]] = i; // a later duplicate wins, as dict(zip(...))
            var ranks = new Dictionary<(string, string), int>();
            for (int i = 0; i < merges.Count; i++) ranks[merges[i]] = i;
            return new ClipBpeTokenizer(encoder, ranks, byteEncoder);
        }

        // bytes_to_unicode's byte order: the printable ranges first ('!'..'~', '¡'..'¬', '®'..'ÿ'), then the rest.
        private static List<int> ByteOrder()
        {
            var order = new List<int>(256);
            for (int b = '!'; b <= '~'; b++) order.Add(b);
            for (int b = 0xA1; b <= 0xAC; b++) order.Add(b);
            for (int b = 0xAE; b <= 0xFF; b++) order.Add(b);
            for (int b = 0; b < 256; b++)
                if (!order.Contains(b)) order.Add(b);
            return order;
        }

        // bytes_to_unicode: printable bytes map to themselves, the others to chr(256 + n) in byte order.
        private static char[] BytesToUnicode()
        {
            var map = new char[256];
            var printable = new HashSet<int>();
            for (int b = '!'; b <= '~'; b++) printable.Add(b);
            for (int b = 0xA1; b <= 0xAC; b++) printable.Add(b);
            for (int b = 0xAE; b <= 0xFF; b++) printable.Add(b);
            int n = 0;
            for (int b = 0; b < 256; b++) map[b] = printable.Contains(b) ? (char)b : (char)(256 + n++);
            return map;
        }

        /// <summary><c>whitespace_clean(basic_clean(text)).lower()</c>, without ftfy (see the remarks).</summary>
        public static string Clean(string text)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            try
            {
                var unescaped = WebUtility.HtmlDecode(WebUtility.HtmlDecode(text)).Trim();
                return Whitespace.Replace(unescaped, " ").Trim().ToLowerInvariant();
            }
            catch (RegexMatchTimeoutException)
            {
                throw new ArgumentException("The text could not be cleaned: pattern matching timed out.", nameof(text));
            }
        }

        /// <summary>The BPE tokens of <paramref name="text"/> after cleaning (no special tokens).</summary>
        public override List<string> Tokenize(string text)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            var tokens = new List<string>();
            MatchCollection pieces;
            try
            {
                pieces = Pattern.Matches(Clean(text));
                _ = pieces.Count;
            }
            catch (RegexMatchTimeoutException)
            {
                throw new ArgumentException("The text could not be tokenized: pattern matching timed out.", nameof(text));
            }

            foreach (Match piece in pieces)
            {
                var bytes = Encoding.UTF8.GetBytes(piece.Value);
                var mapped = new StringBuilder(bytes.Length);
                foreach (var b in bytes) mapped.Append(_byteEncoder[b]);
                tokens.AddRange(Bpe(mapped.ToString()).Split(' '));
            }
            return tokens;
        }

        /// <summary>The token ids of <paramref name="text"/> after cleaning, without special tokens (<c>encode</c>).</summary>
        public List<int> EncodeIds(string text) => ConvertTokensToIds(Tokenize(text));

        /// <summary>
        /// <c>open_clip.tokenize</c>: <c>[StartOfTextId] + EncodeIds(text) + [EndOfTextId]</c>, cut to
        /// <paramref name="contextLength"/> with the last id set to <see cref="EndOfTextId"/>, padded with zeros.
        /// </summary>
        public int[] EncodeContext(string text, int contextLength = 77)
        {
            if (contextLength < 2) throw new ArgumentOutOfRangeException(nameof(contextLength), "The context holds at least the two special tokens.");
            var ids = new List<int> { StartOfTextId };
            ids.AddRange(EncodeIds(text ?? string.Empty));
            ids.Add(EndOfTextId);
            var result = new int[contextLength];
            int count = Math.Min(ids.Count, contextLength);
            for (int i = 0; i < count; i++) result[i] = ids[i];
            if (ids.Count > contextLength) result[contextLength - 1] = EndOfTextId;
            return result;
        }

        /// <summary>
        /// Encodes as <c>open_clip.tokenize</c> does: with special tokens, <c>&lt;start_of_text&gt;</c> first and
        /// <c>&lt;end_of_text&gt;</c> last (kept on truncation to <see cref="EncodingOptions.MaxLength"/>); with padding,
        /// zeros (id 0) to <see cref="EncodingOptions.MaxLength"/> with a zero attention mask.
        /// </summary>
        public override TokenizationResult Encode(string text, EncodingOptions? options = null)
        {
            options ??= new EncodingOptions();
            var ids = EncodeIds(text ?? string.Empty);
            if (options.AddSpecialTokens)
            {
                ids.Insert(0, StartOfTextId);
                ids.Add(EndOfTextId);
            }
            if (options.Truncation && options.MaxLength.HasValue && ids.Count > options.MaxLength.Value)
            {
                int max = options.MaxLength.Value;
                ids = ids.Take(max).ToList();
                if (options.AddSpecialTokens && max > 0) ids[max - 1] = EndOfTextId;
            }
            var mask = Enumerable.Repeat(1, ids.Count).ToList();
            if (options.Padding && options.MaxLength.HasValue && ids.Count < options.MaxLength.Value)
            {
                int padding = options.MaxLength.Value - ids.Count;
                ids.AddRange(Enumerable.Repeat(0, padding));
                mask.AddRange(Enumerable.Repeat(0, padding));
            }
            return new TokenizationResult
            {
                TokenIds = ids,
                Tokens = ConvertIdsToTokens(ids),
                AttentionMask = options.ReturnAttentionMask ? mask : new List<int>()
            };
        }

        // SimpleTokenizer.bpe: repeatedly merge the lowest-ranked adjacent pair of the word (its last symbol carrying
        // </w>), returning the symbols joined by spaces.
        private string Bpe(string token)
        {
            if (_cache.TryGetValue(token, out var cached)) return cached;
            var word = new List<string>(token.Length);
            for (int i = 0; i < token.Length - 1; i++) word.Add(token[i].ToString());
            word.Add(token[token.Length - 1] + WordEnd);
            if (word.Count == 1)
            {
                _cache[token] = word[0];
                return word[0];
            }

            while (word.Count > 1)
            {
                (string, string)? best = null;
                int bestRank = int.MaxValue;
                for (int i = 0; i < word.Count - 1; i++)
                {
                    if (_ranks.TryGetValue((word[i], word[i + 1]), out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        best = (word[i], word[i + 1]);
                    }
                }
                if (best is null) break;
                var (first, second) = best.Value;
                var merged = new List<string>(word.Count);
                for (int i = 0; i < word.Count;)
                {
                    if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
                    {
                        merged.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        merged.Add(word[i]);
                        i++;
                    }
                }
                word = merged;
            }

            var result = string.Join(" ", word);
            _cache[token] = result;
            return result;
        }

        /// <summary>
        /// <c>SimpleTokenizer.decode</c>. With <paramref name="skipSpecialTokens"/>, <c>&lt;start_of_text&gt;</c> is dropped
        /// and the ids stop at the first <c>&lt;end_of_text&gt;</c> (what follows it in a context is zero padding, which
        /// would otherwise decode as <c>"!"</c>); without it every id is decoded.
        /// </summary>
        public override string Decode(List<int> tokenIds, bool skipSpecialTokens = true)
        {
            if (tokenIds is null || tokenIds.Count == 0) return string.Empty;
            var ids = tokenIds;
            if (skipSpecialTokens)
            {
                int end = tokenIds.IndexOf(EndOfTextId);
                ids = (end < 0 ? tokenIds : tokenIds.Take(end)).Where(id => id != StartOfTextId).ToList();
            }
            return CleanupTokens(ConvertIdsToTokens(ids));
        }

        /// <summary><c>SimpleTokenizer.decode</c>: the tokens' bytes as UTF-8, each <c>&lt;/w&gt;</c> a space.</summary>
        protected override string CleanupTokens(List<string> tokens)
        {
            var joined = string.Concat(tokens);
            var bytes = new List<byte>(joined.Length);
            var text = new StringBuilder();
            void Flush()
            {
                if (bytes.Count == 0) return;
                text.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
            }
            for (int i = 0; i < joined.Length; i++)
            {
                if (string.CompareOrdinal(joined, i, WordEnd, 0, WordEnd.Length) == 0)
                {
                    Flush();
                    text.Append(' ');
                    i += WordEnd.Length - 1;
                }
                else if (_byteDecoder.TryGetValue(joined[i], out var b))
                {
                    bytes.Add(b);
                }
            }
            Flush();
            return text.ToString();
        }
    }
}
