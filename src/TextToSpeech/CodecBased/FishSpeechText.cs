using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Fish Speech's text tokenizers: 1.5's tiktoken vocabulary and 1.4's Hugging Face byte-level BPE.</summary>
internal static class FishSpeechText
{
    /// <summary>The 1.5 release's special tokens in id order (<c>ALL_SPECIAL_TOKENS</c>), numbered after the ranks: the
    /// sequence and chat markers, four placeholders, the modality markers, then one <c>&lt;|semantic:i|&gt;</c> per code
    /// (1,024 in the release).</summary>
    public static IReadOnlyList<string> SpecialTokens15(int codes = 1024) => new[]
    {
        "<|begin_of_text|>", "<|end_of_text|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
        "<|placeholder:0|>", "<|placeholder:1|>", "<|placeholder:2|>", "<|placeholder:3|>",
        "<|text|>", "<|voice|>", "<|interleave|>",
    }.Concat(Enumerable.Range(0, codes).Select(i => $"<|semantic:{i}|>")).ToArray();

    /// <summary>The 1.4 release's added tokens, ids 0 to 8 (<c>tokenizer.json</c>'s <c>added_tokens</c>).</summary>
    public static IReadOnlyList<string> SpecialTokens14 { get; } = new[]
    {
        "<|begin_of_sequence|>", "<|end_of_sequence|>", "<|pad|>", "<|im_start|>", "<|im_end|>", "<|semantic|>", "<|mel|>",
        "<|reserve_0|>", "<|reserve_1|>",
    };

    /// <summary>Fish Speech 1.5's tokenizer (<c>FishTokenizer</c>) from its <c>tokenizer.tiktoken</c>.</summary>
    public static TiktokenEncoding Load15(string path, int codes = 1024)
    {
        using var reader = new StreamReader(path ?? throw new ArgumentNullException(nameof(path)));
        return new TiktokenEncoding(TiktokenEncoding.ReadRanks(reader), SpecialTokens15(codes), TiktokenPattern.Fish);
    }

    /// <summary>Fish Speech 1.4's tokenizer from its <c>tokenizer.json</c>.</summary>
    public static ByteLevelBpeTokenizer Load14(string path) =>
        ByteLevelBpeTokenizer.FromJson(File.ReadAllText(path ?? throw new ArgumentNullException(nameof(path))));

    // fish_speech/text/clean.py's SYMBOLS_MAPPING: 1.5 maps only the curly single quotes; 1.4 also maps ellipses and curly
    // double quotes, drops brackets, dashes, tildes, middle dots and newlines, and turns colons and semicolons into commas.
    private static Dictionary<char, string> Map(params (int Code, string To)[] entries) => entries.ToDictionary(e => (char)e.Code, e => e.To);

    private static readonly Dictionary<char, string> Symbols15 = Map((0x2018, "'"), (0x2019, "'"));

    private static readonly Dictionary<char, string> Symbols14 = Map(
        ('\n', ""), (0x2026, "."), (0x201C, "'"), (0x201D, "'"), (0x2018, "'"), (0x2019, "'"),
        (0x3010, ""), (0x3011, ""), ('[', ""), (']', ""), (0xFF08, ""), (0xFF09, ""), ('(', ""), (')', ""),
        (0x30FB, ""), (0x00B7, ""), (0x300C, "'"), (0x300D, "'"), (0x300A, "'"), (0x300B, "'"),
        (0x2014, ""), (0xFF5E, ""), ('~', ""), (0xFF1A, ","), (0xFF1B, ","), (';', ","), (':', ","));

    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private static readonly Regex Commas = new("[,]{2,}", RegexOptions.None, RegexTimeout);
    private static readonly Regex PeriodsAndCommas = new("[.,]{2,}", RegexOptions.None, RegexTimeout);

    // EMOJI_REGEX's ranges: emoticons, symbols and pictographs, transport and map symbols, regional indicators.
    private static bool IsEmoji(int c) => c is >= 0x1F600 and <= 0x1F64F or >= 0x1F300 and <= 0x1F5FF or >= 0x1F680 and <= 0x1F6FF
        or >= 0x1F1E0 and <= 0x1F1FF;

    /// <summary>Python's <c>str.isspace</c>: .NET's white space and the four information separators.</summary>
    internal static bool IsPythonSpace(char c) => char.IsWhiteSpace(c) || (c >= (char)0x1C && c <= (char)0x1F);

    private static string PythonStrip(string text)
    {
        int start = 0, end = text.Length;
        while (start < end && IsPythonSpace(text[start])) start++;
        while (end > start && IsPythonSpace(text[end - 1])) end--;
        return text.Substring(start, end - start);
    }

    // A string's code points as strings (a surrogate pair stays whole), as Python iterates a str.
    private static IEnumerable<string> CodePoints(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                yield return text.Substring(i, 2);
                i++;
            }
            else yield return text[i].ToString();
        }
    }

    /// <summary><c>clean_text</c> of the release: stripped, its symbols mapped, emoji removed, and runs of commas (1.4:
    /// of periods and commas) cut to their first character.</summary>
    public static string Clean(string text, FishSpeechVersion version)
    {
        var symbols = version == FishSpeechVersion.V15 ? Symbols15 : Symbols14;
        var builder = new StringBuilder();
        foreach (var c in PythonStrip(text))
            if (symbols.TryGetValue(c, out var replacement)) builder.Append(replacement);
            else builder.Append(c);
        var kept = new StringBuilder();
        foreach (var point in CodePoints(builder.ToString()))
            if (!IsEmoji(char.ConvertToUtf32(point, 0))) kept.Append(point);
        return (version == FishSpeechVersion.V15 ? Commas : PeriodsAndCommas).Replace(kept.ToString(), m => m.Value.Substring(0, 1));
    }

    private static int Utf8Length(string text) => Encoding.UTF8.GetByteCount(text);

    // break_text: a piece longer than the limit is cut after each of the split characters.
    private static IEnumerable<string> BreakText(IEnumerable<string> texts, int length, ISet<string> splits)
    {
        foreach (var text in texts)
        {
            if (Utf8Length(text) <= length)
            {
                yield return text;
                continue;
            }
            var current = new StringBuilder();
            foreach (var c in CodePoints(text))
            {
                current.Append(c);
                if (splits.Contains(c))
                {
                    yield return current.ToString();
                    current.Clear();
                }
            }
            if (current.Length > 0) yield return current.ToString();
        }
    }

    // break_text_by_length: a piece still too long is cut whenever it reaches the limit.
    private static IEnumerable<string> BreakTextByLength(IEnumerable<string> texts, int length)
    {
        foreach (var text in texts)
        {
            if (Utf8Length(text) <= length)
            {
                yield return text;
                continue;
            }
            var current = new StringBuilder();
            foreach (var c in CodePoints(text))
            {
                current.Append(c);
                if (Utf8Length(current.ToString()) >= length)
                {
                    yield return current.ToString();
                    current.Clear();
                }
            }
            if (current.Length > 0) yield return current.ToString();
        }
    }

    // string.punctuation.
    private const string AsciiPunctuation = @"!""#$%&'()*+,-./:;<=>?@[\]^_`{|}~";

    // add_cleaned: the stripped piece, unless it is only white space and ASCII punctuation.
    private static void AddCleaned(string current, List<string> segments)
    {
        current = PythonStrip(current);
        if (current.Length > 0 && !current.All(c => IsPythonSpace(c) || AsciiPunctuation.IndexOf(c) >= 0)) segments.Add(current);
    }

    private static readonly Regex Float = new(@"(\d+)\.(\d+)", RegexOptions.None, RegexTimeout);
    private static readonly Regex ProtectedFloat = new(@"<(\d+)_f_(\d+)>", RegexOptions.None, RegexTimeout);

    private static readonly HashSet<string> SentenceEnds = new(new[] { ".", "!", "?" }.Concat(new[] { 0x3002, 0xFF01, 0xFF1F }.Select(c => ((char)c).ToString())));
    private static readonly HashSet<string> CommaSplits = new() { ",", ((char)0xFF0C).ToString() };
    private static readonly HashSet<string> SpaceSplits = new() { " " };

    /// <summary>
    /// <c>split_text</c>: the cleaned text cut after sentence ends (not inside a decimal number), then commas, then spaces,
    /// then anywhere, wherever a piece is longer than <paramref name="length"/> UTF-8 bytes, and the pieces merged greedily
    /// into segments of at most that length.
    /// </summary>
    public static List<string> Split(string text, int length, FishSpeechVersion version)
    {
        IEnumerable<string> texts = new[] { Float.Replace(Clean(text, version), "<$1_f_$2>") };
        texts = BreakText(texts, length, SentenceEnds);
        texts = texts.Select(t => ProtectedFloat.Replace(t, "$1.$2"));
        texts = BreakText(texts, length, CommaSplits);
        texts = BreakText(texts, length, SpaceSplits);
        var segments = new List<string>();
        string current = "";
        foreach (var piece in BreakTextByLength(texts, length))
        {
            if (Utf8Length(current) + Utf8Length(piece) <= length) current += piece;
            else
            {
                AddCleaned(current, segments);
                current = piece;
            }
        }
        if (current.Length > 0) AddCleaned(current, segments);
        return segments;
    }
}

/// <summary>A Fish Speech model's text tokenizer: the release's, or a byte-level vocabulary for training from scratch.</summary>
internal sealed class FishTextTokenizer
{
    private readonly Func<string, int[]> _encode;
    private readonly Func<string, int> _tokenId;
    private readonly Func<IEnumerable<int>, string> _decode;

    private FishTextTokenizer(Func<string, int[]> encode, Func<string, int> tokenId, Func<IEnumerable<int>, string> decode, int size, bool release)
    {
        _encode = encode;
        _tokenId = tokenId;
        _decode = decode;
        Size = size;
        IsRelease = release;
    }

    /// <summary>The ids it uses: one past the largest.</summary>
    public int Size { get; }

    /// <summary>Whether this is a release's vocabulary (rather than the byte-level one).</summary>
    public bool IsRelease { get; }

    /// <summary>Encodes text, special tokens recognized.</summary>
    public int[] Encode(string text) => _encode(text);

    /// <summary>A special token's id.</summary>
    public int TokenId(string token) => _tokenId(token);

    /// <summary>Decodes ids to text.</summary>
    public string Decode(IEnumerable<int> ids) => _decode(ids);

    /// <summary>The release's tokenizer from its file (1.5: <c>tokenizer.tiktoken</c>; 1.4: <c>tokenizer.json</c>).</summary>
    public static FishTextTokenizer Release(FishSpeechVersion version, string path, int codes)
    {
        if (version == FishSpeechVersion.V15)
        {
            var t = FishSpeechText.Load15(path, codes);
            return new FishTextTokenizer(text => t.Encode(text), t.SpecialTokenId, t.Decode, t.SpecialTokenId($"<|semantic:{codes - 1}|>") + 1, true);
        }
        var b = FishSpeechText.Load14(path);
        return new FishTextTokenizer(b.Encode, b.TokenId, ids => b.Decode(ids), b.VocabularySize, true);
    }

    /// <summary>
    /// A byte-level vocabulary for a model trained from scratch: each UTF-8 byte is a token (no merges) and the release's
    /// special tokens keep their places relative to it (1.5: after the 256 bytes; 1.4: ids 0 to 8, the bytes after).
    /// </summary>
    public static FishTextTokenizer Bytes(FishSpeechVersion version, int codes)
    {
        if (version == FishSpeechVersion.V15)
        {
            var ranks = new Dictionary<byte[], int>();
            for (int b = 0; b < 256; b++) ranks[new[] { (byte)b }] = b;
            var t = new TiktokenEncoding(ranks, FishSpeechText.SpecialTokens15(codes), TiktokenPattern.Fish);
            return new FishTextTokenizer(text => t.Encode(text), t.SpecialTokenId, t.Decode, 256 + FishSpeechText.SpecialTokens15(codes).Count, false);
        }
        var specials = FishSpeechText.SpecialTokens14;
        var vocabulary = BpeTokenizer.BuildByteToUnicode().ToDictionary(p => p.Value.ToString(), p => specials.Count + p.Key, StringComparer.Ordinal);
        var tokenizer = new ByteLevelBpeTokenizer(vocabulary, new Dictionary<string, int>(), specials.Select((s, i) => (s, i, true)),
            splitPattern: null, nfc: false, ignoreMerges: false);
        return new FishTextTokenizer(tokenizer.Encode, tokenizer.TokenId, ids => tokenizer.Decode(ids), specials.Count + 256, false);
    }
}
