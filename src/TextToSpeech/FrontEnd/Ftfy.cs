using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// ftfy 6.3.1's <c>fix_text</c> with its default configuration (rspeer/python-ftfy, Apache-2.0): it repairs mojibake (text
/// decoded with the wrong codec, detected by its badness heuristic and re-decoded through the codec that explains it),
/// unescapes HTML entities, and fixes C1 controls, ligatures, full-width characters, curly quotes, line breaks,
/// surrogates, terminal escapes and control characters, then normalizes to NFC.
/// </summary>
/// <remarks>The tables ftfy builds at import time (its codecs, HTML entities, full-width map and the badness and UTF-8
/// detector expressions) are exported from ftfy itself by <c>tools/reference-data/ftfy_reference.py</c>.</remarks>
internal static class Ftfy
{
    private sealed class Tables
    {
        public readonly string[] Encodings;
        public readonly Dictionary<string, string> Decode = new(StringComparer.Ordinal);
        public readonly Dictionary<string, Dictionary<char, byte>> Encode = new(StringComparer.Ordinal);
        public readonly Dictionary<string, HashSet<char>> Possible = new(StringComparer.Ordinal);
        public readonly Dictionary<string, string> HtmlEntities = new(StringComparer.Ordinal);
        public readonly Dictionary<int, string> InvalidCharrefs = new();
        public readonly HashSet<int> InvalidCodepoints = new();
        public readonly Dictionary<char, string> WidthMap = new();
        public readonly string PythonWord;
        public readonly string PythonSpace;
        public readonly Regex Badness;
        public readonly Regex Utf8Detector;

        public Tables()
        {
            using var stream = typeof(Ftfy).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.ftfy_tables.json.gz")
                ?? throw new InvalidDataException("The embedded resource ftfy_tables.json.gz is missing.");
            using var gzip = new GZipStream(stream, CompressionMode.Decompress);
            using var reader = new StreamReader(gzip, Encoding.UTF8);
            var root = JObject.Parse(reader.ReadToEnd());
            Encodings = root["charmap_encodings"]!.Select(t => (string)t!).ToArray();
            foreach (var codec in (JObject)root["codecs"]!)
            {
                string table = (string)codec.Value!;
                Decode[codec.Key] = table;
                var encode = new Dictionary<char, byte>();
                for (int b = 0; b < 256; b++)
                    if (!encode.ContainsKey(table[b])) encode[table[b]] = (byte)b;
                Encode[codec.Key] = encode;
                // ENCODING_REGEXES: \x00-\x19, \x1b-\x7f and the characters bytes 128-255 and 26 decode to.
                var possible = new HashSet<char>();
                for (int c = 0; c <= 0x7f; c++) if (c != 0x1a) possible.Add((char)c);
                for (int b = 128; b < 256; b++) possible.Add(table[b]);
                possible.Add(table[26]);
                Possible[codec.Key] = possible;
            }
            foreach (var entity in (JObject)root["html_entities"]!) HtmlEntities[entity.Key] = (string)entity.Value!;
            foreach (var entry in (JObject)root["invalid_charrefs"]!) InvalidCharrefs[int.Parse(entry.Key, CultureInfo.InvariantCulture)] = (string)entry.Value!;
            foreach (var code in root["invalid_codepoints"]!) InvalidCodepoints.Add((int)code);
            foreach (var entry in (JObject)root["width_map"]!) WidthMap[(char)int.Parse(entry.Key, CultureInfo.InvariantCulture)] = (string)entry.Value!;
            PythonWord = (string)root["python_word"]!;
            PythonSpace = (string)root["python_space"]!;
            Badness = new Regex((string)root["badness"]!, RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant);
            Utf8Detector = new Regex((string)root["utf8_detector"]!, RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant);
        }
    }

    private static readonly Lazy<Tables> Shared = new(() => new Tables(), isThreadSafe: true);
    private static Tables Data => Shared.Value;

    private const int MaxDecodeLength = 1000000;

    /// <summary><c>ftfy.fix_text(text)</c>.</summary>
    public static string FixText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var output = new StringBuilder(text.Length);
        bool unescapeHtml = true;   // "auto": off for good once a segment holds '<'
        int position = 0;
        while (position < text.Length)
        {
            int lineBreak = text.IndexOf('\n', position) + 1;
            if (lineBreak == 0) lineBreak = text.Length;
            if (lineBreak - position > MaxDecodeLength) lineBreak = position + MaxDecodeLength;
            string segment = text.Substring(position, lineBreak - position);
            if (unescapeHtml && segment.IndexOf('<') >= 0) unescapeHtml = false;
            output.Append(FixSegment(segment, unescapeHtml));
            position = lineBreak;
        }
        return output.ToString();
    }

    // fix_and_explain: the fixers in order until the text stops changing.
    private static string FixSegment(string text, bool unescapeHtml)
    {
        if (unescapeHtml && text.IndexOf('<') >= 0) unescapeHtml = false;
        while (true)
        {
            string original = text;
            if (unescapeHtml) text = UnescapeHtml(text);
            text = FixEncoding(text);
            text = FixC1Controls(text);
            text = FixLatinLigatures(text);
            text = FixCharacterWidth(text);
            text = UncurlQuotes(text);
            text = FixLineBreaks(text);
            text = FixSurrogates(text);
            text = RemoveTerminalEscapes(text);
            text = RemoveControlChars(text);
            text = text.Normalize(NormalizationForm.FormC);
            if (text == original) return text;
        }
    }

    /// <summary><c>ftfy.fix_encoding</c>: one re-decoding step at a time until the text stops changing.</summary>
    public static string FixEncoding(string text)
    {
        while (true)
        {
            string previous = text;
            text = FixEncodingOneStep(text);
            if (text == previous) return text;
        }
    }

    /// <summary>Python's <c>\w</c> and <c>\s</c> as .NET character classes (<c>str.isalnum()</c> or '_', and
    /// <c>str.isspace()</c>; the word class holds the Basic Multilingual Plane only).</summary>
    internal static (string Word, string Space) PythonClasses => (Data.PythonWord, Data.PythonSpace);

    internal static bool IsBad(string text) => Data.Badness.IsMatch(text);

    private static bool PossibleEncoding(string text, string encoding)
    {
        if (encoding == "ascii") return text.All(c => c <= 0x7f);
        var set = Data.Possible[encoding];
        return text.All(set.Contains);
    }

    private static string FixEncodingOneStep(string text)
    {
        if (text.Length == 0) return text;
        if (PossibleEncoding(text, "ascii") || !IsBad(text)) return text;
        var possible = new List<string>();
        foreach (var encoding in Data.Encodings)
        {
            if (!PossibleEncoding(text, encoding)) continue;
            possible.Add(encoding);
            var bytes = EncodeCharmap(text, encoding);
            if (encoding != "macroman" && AlteredUtf8.IsMatch(Latin1(bytes))) bytes = RestoreByteA0(bytes);
            if (encoding.StartsWith("sloppy", StringComparison.Ordinal)) bytes = ReplaceLossySequences(bytes);
            bool variants = Array.IndexOf(bytes, (byte)237) >= 0 || Array.IndexOf(bytes, (byte)192) >= 0;
            var decoded = variants ? DecodeUtf8Variants(bytes) : DecodeUtf8(bytes);
            if (decoded is not null) return decoded;
        }
        if (Data.Utf8Detector.IsMatch(text))
        {
            string fixed1 = DecodeInconsistentUtf8(text);
            if (fixed1 != text) return fixed1;
        }
        // ftfy tests for "windows-1252", which is not among its codecs (only the sloppy one is), so a latin-1 text is
        // always tried as latin-1 bytes decoded by strict windows-1252.
        if (possible.Contains("latin-1"))
        {
            var bytes = EncodeCharmap(text, "latin-1");
            if (!bytes.Any(b => Array.IndexOf(Windows1252Undefined, b) >= 0))
            {
                var table = Data.Decode["sloppy-windows-1252"];
                string fixed2 = new(bytes.Select(b => b == 26 ? '\u001a' : table[b]).ToArray());
                if (fixed2 != text) return fixed2;
            }
        }
        if (C1Control.IsMatch(text)) return FixC1Controls(text);
        return text;
    }

    // ---------------------------------------------------------------- codecs

    // The bytes windows-1252 leaves undefined (its sloppy variant reads them as Latin-1).
    private static readonly byte[] Windows1252Undefined = { 0x81, 0x8d, 0x8f, 0x90, 0x9d };

    private static byte[] EncodeCharmap(string text, string encoding)
    {
        var map = Data.Encode[encoding];
        var bytes = new byte[text.Length];
        for (int i = 0; i < text.Length; i++) bytes[i] = map[text[i]];
        return bytes;
    }

    private static string Latin1(byte[] bytes)
    {
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) chars[i] = (char)bytes[i];
        return new string(chars);
    }

    private static byte[] FromLatin1(string text)
    {
        var bytes = new byte[text.Length];
        for (int i = 0; i < text.Length; i++) bytes[i] = (byte)text[i];
        return bytes;
    }

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    // bytes.decode("utf-8"), null when Python raises UnicodeDecodeError.
    private static string? DecodeUtf8(byte[] bytes)
    {
        try
        {
            return StrictUtf8.GetString(bytes);
        }
        catch (DecoderFallbackException)
        {
            return null;
        }
    }

    private static readonly Regex SpecialBytes = new(
        @"(\xc0(\x80|$))|(\xed([\xa0-\xaf]|$)([\x80-\xbf]|$)(\xed|$)([\xb0-\xbf]|$)([\x80-\xbf]|$))|(\xed([\xa0-\xbf]|$)([\x80-\xbf]|$))",
        RegexOptions.CultureInvariant);
    private static readonly Regex Cesu8 = new(@"^\xed[\xa0-\xaf][\x80-\xbf]\xed[\xb0-\xbf][\x80-\xbf]", RegexOptions.CultureInvariant);

    // ftfy's "utf-8-variants": UTF-8 that also reads Java's C0 80 as NUL and CESU-8's surrogate pairs; null on a decode
    // error.
    private static string? DecodeUtf8Variants(byte[] bytes)
    {
        var output = new StringBuilder();
        int position = 0;
        while (position < bytes.Length)
        {
            var rest = new byte[bytes.Length - position];
            Array.Copy(bytes, position, rest, 0, rest.Length);
            var match = SpecialBytes.Match(Latin1(rest));
            if (!match.Success)
            {
                var tail = DecodeUtf8(rest);
                if (tail is null) return null;
                output.Append(tail);
                break;
            }
            if (match.Index > 0)
            {
                var prefix = new byte[match.Index];
                Array.Copy(rest, prefix, prefix.Length);
                var decoded = DecodeUtf8(prefix);
                if (decoded is null) return null;
                output.Append(decoded);
                position += prefix.Length;
                continue;
            }
            if (rest[0] == 0xc0)
            {
                if (rest.Length > 1)
                {
                    output.Append('\0');
                    position += 2;
                    continue;
                }
                return null;
            }
            if (rest.Length >= 6 && Cesu8.IsMatch(Latin1(rest)))
            {
                int codepoint = ((rest[1] & 15) << 16) + ((rest[2] & 63) << 10) + ((rest[4] & 15) << 6) + (rest[5] & 63) + 65536;
                output.Append(char.ConvertFromUtf32(codepoint));
                position += 6;
                continue;
            }
            return null;   // an encoded surrogate outside a CESU-8 pair: strict UTF-8 rejects it
        }
        return output.ToString();
    }

    // ---------------------------------------------------------------- byte-level repairs

    private static readonly Regex AlteredUtf8 = new(
        @"[\xc2\xc3\xc5\xce\xd0\xd9][ ]|[\xe2\xe3][ ][\x80-\x84\x86-\x9f\xa1-\xbf]|[\xe0-\xe3][\x80-\x84\x86-\x9f\xa1-\xbf][ ]|[\xf0][ ][\x80-\xbf][\x80-\xbf]|[\xf0][\x80-\xbf][ ][\x80-\xbf]|[\xf0][\x80-\xbf][\x80-\xbf][ ]",
        RegexOptions.CultureInvariant);
    private static readonly Regex LossyUtf8 = new(
        @"[\xc2-\xdf][\x1a]|[\xc2-\xc3][?]|\xed[\xa0-\xaf][\x1a?]\xed[\xb0-\xbf][\x1a?\x80-\xbf]|\xed[\xa0-\xaf][\x1a?\x80-\xbf]\xed[\xb0-\xbf][\x1a?]|[\xe0-\xef][\x1a?][\x1a\x80-\xbf]|[\xe0-\xef][\x1a\x80-\xbf][\x1a?]|[\xf0-\xf4][\x1a?][\x1a\x80-\xbf][\x1a\x80-\xbf]|[\xf0-\xf4][\x1a\x80-\xbf][\x1a?][\x1a\x80-\xbf]|[\xf0-\xf4][\x1a\x80-\xbf][\x1a\x80-\xbf][\x1a?]|\x1a",
        RegexOptions.CultureInvariant);
    private static readonly Regex AGraveWord = new(@"\xc3 (?! |quele|quela|quilo|s )", RegexOptions.CultureInvariant);

    // restore_byte_a0: a space that should be the byte A0 inside a UTF-8 sequence.
    private static byte[] RestoreByteA0(byte[] bytes)
    {
        string text = AGraveWord.Replace(Latin1(bytes), "\u00c3\u00a0 ");
        text = AlteredUtf8.Replace(text, m => m.Value.Replace(' ', '\u00a0'));
        return FromLatin1(text);
    }

    // replace_lossy_sequences: sequences a sloppy codec lost bytes from become U+FFFD's UTF-8 bytes.
    private static byte[] ReplaceLossySequences(byte[] bytes) => FromLatin1(LossyUtf8.Replace(Latin1(bytes), "\u00ef\u00bf\u00bd"));

    private static string DecodeInconsistentUtf8(string text) =>
        Data.Utf8Detector.Replace(text, m => m.Value.Length < text.Length && IsBad(m.Value) ? FixEncoding(m.Value) : m.Value);

    // ---------------------------------------------------------------- the character fixers

    private static readonly Regex C1Control = new("[\u0080-\u009f]", RegexOptions.CultureInvariant);

    private static string FixC1Controls(string text) =>
        C1Control.Replace(text, m => Data.Decode["sloppy-windows-1252"][m.Value[0]].ToString());

    private static readonly Regex HtmlEntity = new("&#?[0-9A-Za-z]{1,24};", RegexOptions.CultureInvariant);

    private static string UnescapeHtml(string text) => HtmlEntity.Replace(text, m =>
    {
        string entity = m.Value;
        if (Data.HtmlEntities.TryGetValue(entity, out var value)) return value;
        if (entity.StartsWith("&#", StringComparison.Ordinal))
        {
            string unescaped = HtmlUnescapeNumeric(entity);
            return unescaped.IndexOf(';') >= 0 ? entity : unescaped;
        }
        return entity;
    });

    private static readonly Regex Charref = new("&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?)", RegexOptions.CultureInvariant);

    // html.unescape on a numeric reference (its _replace_charref); anything it does not read stays as written.
    private static string HtmlUnescapeNumeric(string text) => Charref.Replace(text, m =>
    {
        string s = m.Groups[1].Value.TrimEnd(';');
        bool hex = s.Length > 1 && (s[1] == 'x' || s[1] == 'X');
        string digits = hex ? s.Substring(2) : s.Substring(1);
        long number = 0;
        bool huge = false;
        foreach (char c in digits)
        {
            int d = hex ? Convert.ToInt32(c.ToString(), 16) : c - '0';
            number = number * (hex ? 16 : 10) + d;
            if (number > 0x10FFFF) { huge = true; break; }
        }
        if (!huge && Data.InvalidCharrefs.TryGetValue((int)number, out var mapped)) return mapped;
        if (huge || (number >= 0xD800 && number <= 0xDFFF)) return "\ufffd";
        if (Data.InvalidCodepoints.Contains((int)number)) return "";
        return char.ConvertFromUtf32((int)number);
    });

    private static readonly Regex Ansi = new(@"\x1b\[((?:\d|;)*)([a-zA-Z])", RegexOptions.CultureInvariant);

    private static string RemoveTerminalEscapes(string text) => Ansi.Replace(text, "");

    private static readonly Regex SingleQuote = new("[\u02bc\u2018-\u201b]", RegexOptions.CultureInvariant);
    private static readonly Regex DoubleQuote = new("[\u201c-\u201f]", RegexOptions.CultureInvariant);

    private static string UncurlQuotes(string text) => SingleQuote.Replace(DoubleQuote.Replace(text, "\""), "'");

    private static readonly Dictionary<char, string> Ligatures = new()
    {
        ['\u0132'] = "IJ", ['\u0133'] = "ij", ['\u0149'] = "\u02bcn", ['\u01f1'] = "DZ", ['\u01f2'] = "Dz", ['\u01f3'] = "dz",
        ['\u01c4'] = "D\u017d", ['\u01c5'] = "D\u017e", ['\u01c6'] = "d\u017e", ['\u01c7'] = "LJ", ['\u01c8'] = "Lj", ['\u01c9'] = "lj",
        ['\u01ca'] = "NJ", ['\u01cb'] = "Nj", ['\u01cc'] = "nj", ['\ufb00'] = "ff", ['\ufb01'] = "fi", ['\ufb02'] = "fl",
        ['\ufb03'] = "ffi", ['\ufb04'] = "ffl", ['\ufb05'] = "\u017ft", ['\ufb06'] = "st",
    };

    private static string Translate(string text, Func<char, string?> map)
    {
        StringBuilder? output = null;
        for (int i = 0; i < text.Length; i++)
        {
            string? replacement = map(text[i]);
            if (replacement is null)
            {
                output?.Append(text[i]);
                continue;
            }
            output ??= new StringBuilder(text, 0, i, text.Length);
            output.Append(replacement);
        }
        return output?.ToString() ?? text;
    }

    private static string FixLatinLigatures(string text) => Translate(text, c => Ligatures.TryGetValue(c, out var r) ? r : null);

    private static string FixCharacterWidth(string text) => Translate(text, c => Data.WidthMap.TryGetValue(c, out var r) ? r : null);

    private static string FixLineBreaks(string text) =>
        text.Replace("\r\n", "\n").Replace("\r", "\n").Replace("\u2028", "\n").Replace("\u2029", "\n").Replace("\u0085", "\n");

    // fix_surrogates: a surrogate outside a pair becomes U+FFFD (a pair is already one character here).
    private static string FixSurrogates(string text)
    {
        StringBuilder? output = null;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            bool paired = char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]);
            if (paired)
            {
                output?.Append(c).Append(text[i + 1]);
                i++;
                continue;
            }
            if (char.IsSurrogate(c))
            {
                output ??= new StringBuilder(text, 0, i, text.Length);
                output.Append('\ufffd');
                continue;
            }
            output?.Append(c);
        }
        return output?.ToString() ?? text;
    }

    private static bool IsControl(char c) =>
        c <= 8 || c == 11 || (c >= 14 && c <= 31) || c == 127 || (c >= 0x206a && c <= 0x206f) || c == 0xfeff || (c >= 0xfff9 && c <= 0xfffc);

    private static string RemoveControlChars(string text) => Translate(text, c => IsControl(c) ? "" : null);
}
