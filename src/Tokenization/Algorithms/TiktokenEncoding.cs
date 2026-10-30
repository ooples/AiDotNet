using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>The pre-tokenizer pattern a <see cref="TiktokenEncoding"/> splits text with.</summary>
    internal enum TiktokenPattern
    {
        /// <summary>GPT-2's (r50k, Whisper): <c>'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+</c>.</summary>
        Gpt2,

        /// <summary>cl100k's, as GLM-4 uses it: <c>(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
        /// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+</c>.</summary>
        Cl100k,

        /// <summary>Fish Speech 1.5's (<c>FISH_TIKTOKEN_PATTERN</c>): <c>(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{P}|[^\r\n\p{L}\p{N}]?\p{L}+|
        /// \p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(\?!\S)|\s+</c>: punctuation and digits one at a time, and the release's
        /// <c>(\?!\S)</c>, a literal "?!" rather than a lookahead.</summary>
        Fish,
    }

    /// <summary>
    /// A tiktoken byte-level BPE encoding (OpenAI's <c>tiktoken.Encoding</c>): a rank for every mergeable byte sequence,
    /// the GPT-2 pre-tokenizer pattern, and special tokens numbered after the ranks.
    /// </summary>
    /// <remarks>
    /// <para>Unlike merges-list BPE, tiktoken has no merge table: within a pre-token it repeatedly joins the adjacent
    /// pair whose concatenation has the lowest rank (the leftmost on ties) until no adjacent pair is a token, as
    /// <c>_byte_pair_merge</c> does, and each final part's rank is its id.</para>
    /// <para>The pattern <c>'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+</c> is matched by
    /// a scanner over code points rather than a .NET regex: .NET classifies each half of a surrogate pair as
    /// <c>Cs</c>, so a supplementary-plane letter (𠀀) would fall in the "other symbols" branch where tiktoken's Rust
    /// regex sees a letter.</para>
    /// </remarks>
    internal sealed class TiktokenEncoding
    {
        // Byte sequences are keyed as strings with one char per byte (0–255), which hash and compare by value.
        private readonly Dictionary<string, int> _ranks;
        private readonly string[] _bytesById;
        private readonly Dictionary<string, int> _specials;
        private readonly Dictionary<int, string> _specialById;
        private readonly Regex? _specialPattern;
        private readonly TiktokenPattern _pattern;

        /// <param name="ranks">Each mergeable byte sequence's rank; the ranks must be exactly 0 … count − 1.</param>
        /// <param name="specialTokens">Special tokens in id order, numbered from the number of ranks.</param>
        /// <param name="pattern">The pre-tokenizer pattern (GPT-2's).</param>
        public TiktokenEncoding(IReadOnlyDictionary<byte[], int> ranks, IReadOnlyList<string> specialTokens,
            TiktokenPattern pattern = TiktokenPattern.Gpt2)
        {
            _pattern = pattern;
            if (ranks is null) throw new ArgumentNullException(nameof(ranks));
            if (specialTokens is null) throw new ArgumentNullException(nameof(specialTokens));
            _ranks = new Dictionary<string, int>(ranks.Count, StringComparer.Ordinal);
            _bytesById = new string[ranks.Count];
            foreach (var pair in ranks)
            {
                if (pair.Value < 0 || pair.Value >= ranks.Count || _bytesById[pair.Value] is not null)
                    throw new InvalidDataException($"The ranks are not exactly 0 … {ranks.Count - 1}: rank {pair.Value} is out of place.");
                string key = ToKey(pair.Key, 0, pair.Key.Length);
                _ranks.Add(key, pair.Value);
                _bytesById[pair.Value] = key;
            }
            _specials = new Dictionary<string, int>(StringComparer.Ordinal);
            _specialById = new Dictionary<int, string>();
            foreach (var token in specialTokens)
            {
                int id = ranks.Count + _specials.Count;
                _specials.Add(token, id);
                _specialById.Add(id, token);
            }
            if (_specials.Count > 0)
                _specialPattern = new Regex(string.Join("|", _specials.Keys.Select(Regex.Escape)), RegexOptions.CultureInvariant);
        }

        /// <summary>Reads a <c>.tiktoken</c> file into an encoding with <paramref name="specialTokens"/>.</summary>
        public static TiktokenEncoding Load(TextReader reader, IReadOnlyList<string> specialTokens) =>
            new(ReadRanks(reader), specialTokens);

        /// <summary>Reads the ranks of a <c>.tiktoken</c> file: one <c>base64-bytes rank</c> pair per line.</summary>
        public static Dictionary<byte[], int> ReadRanks(TextReader reader)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            var ranks = new Dictionary<byte[], int>();
            string? line;
            int number = 0;
            while ((line = reader.ReadLine()) is not null)
            {
                number++;
                if (line.Length == 0) continue;
                int space = line.IndexOf(' ');
                if (space <= 0 || !int.TryParse(line.Substring(space + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int rank))
                    throw new InvalidDataException($"Line {number} of the .tiktoken file is not 'base64 rank'.");
                // Whisper's multilingual.tiktoken ends with "= 50256": Python's b64decode reads padding alone as no
                // bytes, so that rank is the empty sequence (no pre-token can reach it) and the specials start after it.
                string encoded = line.Substring(0, space);
                ranks.Add(encoded.Trim('=').Length == 0 ? Array.Empty<byte>() : Convert.FromBase64String(encoded), rank);
            }
            return ranks;
        }

        /// <summary>The number of ids: the ranks, then the special tokens.</summary>
        public int VocabularySize => _bytesById.Length + _specials.Count;

        /// <summary>The special tokens and their ids.</summary>
        public IReadOnlyDictionary<string, int> SpecialTokens => _specials;

        /// <summary>The id of special token <paramref name="token"/>.</summary>
        public int SpecialTokenId(string token) =>
            _specials.TryGetValue(token, out int id) ? id : throw new KeyNotFoundException($"'{token}' is not a special token of this encoding.");

        /// <summary>
        /// The ids of <paramref name="text"/>. With <paramref name="allowSpecial"/> (tiktoken's <c>allowed_special="all"</c>)
        /// special tokens written in the text become their ids; without it they are encoded as ordinary text.
        /// </summary>
        public int[] Encode(string text, bool allowSpecial = true)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            var ids = new List<int>();
            int start = 0;
            if (allowSpecial && _specialPattern is not null)
            {
                for (var match = _specialPattern.Match(text); match.Success; match = match.NextMatch())
                {
                    EncodeOrdinary(text, start, match.Index, ids);
                    ids.Add(_specials[match.Value]);
                    start = match.Index + match.Length;
                }
            }
            EncodeOrdinary(text, start, text.Length, ids);
            return ids.ToArray();
        }

        /// <summary>The text of <paramref name="ids"/>, their bytes decoded as UTF-8 (invalid sequences replaced).</summary>
        public string Decode(IEnumerable<int> ids)
        {
            if (ids is null) throw new ArgumentNullException(nameof(ids));
            var bytes = new List<byte>();
            foreach (int id in ids)
            {
                if (id >= 0 && id < _bytesById.Length)
                    foreach (char c in _bytesById[id]) bytes.Add((byte)c);
                else if (_specialById.TryGetValue(id, out var special))
                    bytes.AddRange(Encoding.UTF8.GetBytes(special));
                else
                    throw new ArgumentOutOfRangeException(nameof(ids), id, $"Token ids run 0 … {VocabularySize - 1}.");
            }
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        /// <summary>The pre-tokens of ordinary text (special tokens not matched), as the pattern splits it.</summary>
        internal List<string> PreTokens(string text)
        {
            var pieces = new List<string>();
            for (int position = 0; position < text.Length;)
            {
                int length = _pattern == TiktokenPattern.Cl100k ? Cl100kTokenLength(text, position, text.Length)
                    : _pattern == TiktokenPattern.Fish ? FishTokenLength(text, position, text.Length) : PreTokenLength(text, position, text.Length);
                pieces.Add(text.Substring(position, length));
                position += length;
            }
            return pieces;
        }

        private void EncodeOrdinary(string text, int start, int end, List<int> ids)
        {
            int position = start;
            while (position < end)
            {
                int length = _pattern == TiktokenPattern.Cl100k ? Cl100kTokenLength(text, position, end)
                    : _pattern == TiktokenPattern.Fish ? FishTokenLength(text, position, end) : PreTokenLength(text, position, end);
                var utf8 = Encoding.UTF8.GetBytes(text.Substring(position, length));
                string piece = ToKey(utf8, 0, utf8.Length);
                if (_ranks.TryGetValue(piece, out int whole)) ids.Add(whole);
                else BytePairMerge(piece, ids);
                position += length;
            }
        }

        // tiktoken's _byte_pair_merge. parts[i] is a part's start and the rank of the part joined to its right neighbour.
        private void BytePairMerge(string piece, List<int> ids)
        {
            var starts = new List<int>(piece.Length + 1);
            var pairRanks = new List<int>(piece.Length + 1);
            for (int i = 0; i + 1 < piece.Length; i++)
            {
                starts.Add(i);
                pairRanks.Add(RankOf(piece, i, i + 2));
            }
            starts.Add(piece.Length - 1);
            pairRanks.Add(int.MaxValue);
            starts.Add(piece.Length);
            pairRanks.Add(int.MaxValue);

            int Rank(int i) => i + 3 < starts.Count ? RankOf(piece, starts[i], starts[i + 3]) : int.MaxValue;

            while (true)
            {
                int best = -1, bestRank = int.MaxValue;
                for (int i = 0; i < pairRanks.Count - 1; i++)
                    if (pairRanks[i] < bestRank) { bestRank = pairRanks[i]; best = i; }
                if (best < 0) break;
                if (best > 0) pairRanks[best - 1] = Rank(best - 1);
                pairRanks[best] = Rank(best);
                starts.RemoveAt(best + 1);
                pairRanks.RemoveAt(best + 1);
            }
            for (int i = 0; i + 1 < starts.Count; i++)
            {
                if (!_ranks.TryGetValue(piece.Substring(starts[i], starts[i + 1] - starts[i]), out int id))
                    throw new InvalidDataException("A byte has no rank; a tiktoken vocabulary must rank all 256 single bytes.");
                ids.Add(id);
            }
        }

        private int RankOf(string piece, int from, int to) =>
            _ranks.TryGetValue(piece.Substring(from, to - from), out int rank) ? rank : int.MaxValue;

        private static string ToKey(byte[] bytes, int offset, int count)
        {
            var chars = new char[count];
            for (int i = 0; i < count; i++) chars[i] = (char)bytes[offset + i];
            return new string(chars);
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

        // The end of a run of `kind` starting at `index` (index itself when the run is empty).
        private static int RunEnd(string text, int index, int end, Kind kind)
        {
            while (index < end && KindAt(text, index, out int width) == kind) index += width;
            return index;
        }

        /// <summary>The length of the pattern's match at <paramref name="position"/>, the alternatives tried in order.</summary>
        private static int PreTokenLength(string text, int position, int end)
        {
            // 's|'t|'re|'ve|'m|'ll|'d
            if (text[position] == '\'' && position + 1 < end)
            {
                char a = text[position + 1];
                if (a is 's' or 't' or 'm' or 'd') return 2;
                if (position + 2 < end)
                {
                    char b = text[position + 2];
                    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
                }
            }
            //  ?\p{L}+ |  ?\p{N}+ |  ?[^\s\p{L}\p{N}]+
            foreach (var kind in new[] { Kind.Letter, Kind.Number, Kind.Other })
            {
                int from = text[position] == ' ' && position + 1 < end && KindAt(text, position + 1, out _) == kind ? position + 1 : position;
                if (KindAt(text, from, out _) != kind) continue;
                return RunEnd(text, from, end, kind) - position;
            }
            // \s+(?!\S) | \s+ — a run of whitespace, giving its last character to a following non-space when it can.
            int spaceEnd = RunEnd(text, position, end, Kind.Space);
            int run = spaceEnd - position;
            return spaceEnd == end || run == 1 ? run : run - 1;
        }

        private static bool IsNewline(char c) => c is '\r' or '\n';

        /// <summary>The length of cl100k's pattern match at <paramref name="position"/>, its alternatives in order.</summary>
        private static int FishTokenLength(string text, int position, int end)
        {
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (text[position] == '\'' && position + 1 < end)
            {
                char a = char.ToLowerInvariant(text[position + 1]);
                if (a is 's' or 't' or 'm' or 'd') return 2;
                if (position + 2 < end)
                {
                    char b = char.ToLowerInvariant(text[position + 2]);
                    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
                }
            }
            var kind = KindAt(text, position, out int width);
            // \p{P}: one punctuation character.
            if (IsPunctuation(text, position)) return width;
            // [^\r\n\p{L}\p{N}]?\p{L}+ (the optional character cannot be punctuation, which the branch above took)
            if (kind == Kind.Letter) return RunEnd(text, position, end, Kind.Letter) - position;
            if (kind != Kind.Number && !IsNewline(text[position]) && position + width < end && KindAt(text, position + width, out _) == Kind.Letter)
                return RunEnd(text, position + width, end, Kind.Letter) - position;
            // \p{N}: one digit.
            if (kind == Kind.Number) return width;
            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            int from = text[position] == ' ' && position + 1 < end && KindAt(text, position + 1, out _) == Kind.Other ? position + 1 : position;
            if (KindAt(text, from, out _) == Kind.Other)
            {
                int index = RunEnd(text, from, end, Kind.Other);
                while (index < end && IsNewline(text[index])) index++;
                return index - position;
            }
            // \s*[\r\n]+ : the whitespace run up to and including its last newline.
            int spaceEnd = RunEnd(text, position, end, Kind.Space);
            for (int i = spaceEnd - 1; i >= position; i--)
                if (IsNewline(text[i])) return i + 1 - position;
            // \s+(\?!\S): the run, a literal "?!" and one non-space character.
            if (spaceEnd + 2 < end && text[spaceEnd] == '?' && text[spaceEnd + 1] == '!' && KindAt(text, spaceEnd + 2, out int w) != Kind.Space)
                return spaceEnd + 2 + w - position;
            // \s+
            return spaceEnd - position;
        }

        private static bool IsPunctuation(string text, int index) => CharUnicodeInfo.GetUnicodeCategory(text, index) is UnicodeCategory.ConnectorPunctuation
            or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;

        private static int Cl100kTokenLength(string text, int position, int end)
        {
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (text[position] == '\'' && position + 1 < end)
            {
                char a = char.ToLowerInvariant(text[position + 1]);
                if (a is 's' or 't' or 'm' or 'd') return 2;
                if (position + 2 < end)
                {
                    char b = char.ToLowerInvariant(text[position + 2]);
                    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
                }
            }
            var kind = KindAt(text, position, out int width);
            // [^\r\n\p{L}\p{N}]?\p{L}+
            if (kind == Kind.Letter) return RunEnd(text, position, end, Kind.Letter) - position;
            if (kind != Kind.Number && !IsNewline(text[position]) && position + width < end && KindAt(text, position + width, out _) == Kind.Letter)
                return RunEnd(text, position + width, end, Kind.Letter) - position;
            // \p{N}{1,3}
            if (kind == Kind.Number)
            {
                int index = position;
                for (int n = 0; n < 3 && index < end && KindAt(text, index, out int w) == Kind.Number; n++) index += w;
                return index - position;
            }
            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            int from = text[position] == ' ' && position + 1 < end && KindAt(text, position + 1, out _) == Kind.Other ? position + 1 : position;
            if (KindAt(text, from, out _) == Kind.Other)
            {
                int index = RunEnd(text, from, end, Kind.Other);
                while (index < end && IsNewline(text[index])) index++;
                return index - position;
            }
            // \s*[\r\n]+ : the whitespace run up to and including its last newline.
            int spaceEnd = RunEnd(text, position, end, Kind.Space);
            for (int i = spaceEnd - 1; i >= position; i--)
                if (IsNewline(text[i])) return i + 1 - position;
            // \s+(?!\S) | \s+
            int run = spaceEnd - position;
            return spaceEnd == end || run == 1 ? run : run - 1;
        }
    }
}
