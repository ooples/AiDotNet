using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The part of spaCy 3.6's <c>ru_core_news_sm-3.6.0</c> pipeline that decides each token's part of speech and morphology:
/// the rule-based tokenizer, the hashed-embedding CNN (<c>tok2vec</c>), the <c>morphologizer</c> and the attribute
/// ruler's whitespace rules. The Russian stresser reads these.
/// </summary>
/// <remarks>
/// <para><b>Tokenizer</b> (<c>Tokenizer.__call__</c>): the text splits at whitespace boundaries (a single space between
/// tokens is the previous token's trailing space; other whitespace runs are tokens); each chunk is a special case
/// (the language's exceptions) or has its prefixes and suffixes split off repeatedly, then its infixes; finally special
/// cases that only appear across those pieces are merged back.</para>
/// <para><b>tok2vec</b> (Thinc <c>Tok2Vec.v2</c>): six lexical attributes (norm, first letter, last three letters, shape,
/// the trailing-space flag, the whitespace flag) are hashed (MurmurHash) into embedding tables of 5000, 1000, 2500,
/// 2500, 50 and 50 rows, concatenated, reduced by a maxout layer with layer norm, then four residual blocks of a
/// one-token window, maxout and layer norm. <b>morphologizer</b>: a linear layer whose highest-scoring label gives the
/// token's POS and features.</para>
/// </remarks>
internal sealed class SpacyRussianTagger
{
    /// <summary>A token: its text, whether a space follows it, its NORM when an exception sets it, and its tags.</summary>
    internal sealed class Token
    {
        public string Text = "";
        public bool Space;
        public string? Norm;
        public string Pos = "";
        public string Morph = "";

        /// <summary>The token's text and the space after it (<c>token.text_with_ws</c>).</summary>
        public string TextWithWhitespace => Space ? Text + " " : Text;

        /// <summary><c>token.morph.to_dict()</c>.</summary>
        public Dictionary<string, string> MorphFeatures() => Morph.Length == 0
            ? new Dictionary<string, string>()
            : Morph.Split('|').Select(f => f.Split(new[] { '=' }, 2)).ToDictionary(f => f[0], f => f.Length > 1 ? f[1] : "");
    }

    private static readonly Dictionary<int, string> PosNames = new()
    {
        [84] = "ADJ", [85] = "ADP", [86] = "ADV", [87] = "AUX", [88] = "CONJ", [89] = "CCONJ", [90] = "DET", [91] = "INTJ",
        [92] = "NOUN", [93] = "NUM", [94] = "PART", [95] = "PRON", [96] = "PROPN", [97] = "PUNCT", [98] = "SCONJ", [99] = "SYM",
        [100] = "VERB", [101] = "X", [102] = "EOL", [103] = "SPACE",
    };

    // spacy.lang.norm_exceptions.BASE_NORMS: the lexeme NORM of quotes, dashes and currency signs (else the lower case).
    private static readonly Dictionary<string, string> BaseNorms = new(StringComparer.Ordinal)
    {
        ["'s"] = "'s", ["'S"] = "'s", ["\u2019s"] = "'s", ["\u2019S"] = "'s", ["\u2019"] = "'", ["\u2018"] = "'",
        ["\u00B4"] = "'", ["`"] = "'", ["\u201D"] = "\"", ["\u201C"] = "\"", ["''"] = "\"", ["``"] = "\"", ["\u00B4\u00B4"] = "\"",
        ["\u201E"] = "\"", ["\u00BB"] = "\"", ["\u00AB"] = "\"", ["\u2018\u2018"] = "\"", ["\u2019\u2019"] = "\"",
        ["\uFF1F"] = "?", ["\uFF01"] = "!", ["\uFF0C"] = ",", ["\uFF1B"] = ";", ["\uFF1A"] = ":", ["\u3002"] = ".",
        ["\u0964"] = ".", ["\u2026"] = "...", ["\u2014"] = "-", ["\u2013"] = "-", ["--"] = "-", ["---"] = "-",
        ["\u2014\u2014"] = "-", ["\u20AC"] = "$", ["\u00A3"] = "$", ["\u00A5"] = "$", ["\u0E3F"] = "$", ["US$"] = "$",
        ["C$"] = "$", ["A$"] = "$", ["\u20BA"] = "$", ["\u20B9"] = "$", ["\u09F3"] = "$", ["\u20A9"] = "$", ["Mex$"] = "$",
        ["\u20A3"] = "$", ["E\u00A3"] = "$",
    };

    private readonly Regex? _prefix, _suffix, _infix, _url;
    private readonly Dictionary<string, (string Orth, string? Norm)[]> _specials = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<string[]>> _patterns = new(StringComparer.Ordinal);
    private readonly SpacyTok2Vec _network;
    private readonly string[] _labelMorph;
    private readonly string[] _labelPos;

    private SpacyRussianTagger(Dictionary<object, object?> tokenizer, SpacyTok2Vec network, JObject morphologizerConfig)
    {
        Regex? Compile(string key) => tokenizer.TryGetValue(key, out var p) && p is string pattern
            ? new Regex(PythonPattern(pattern), RegexOptions.CultureInvariant) : null;
        _prefix = Compile("prefix_search");
        _suffix = Compile("suffix_search");
        _infix = Compile("infix_finditer");
        _url = Compile("url_match");
        if (tokenizer.TryGetValue("token_match", out var tokenMatch) && tokenMatch is string)
            throw new NotSupportedException("This pipeline's tokenizer has a token_match rule, which the port does not read.");
        _network = network;
        var morph = (JObject)morphologizerConfig["labels_morph"]!;
        var pos = (JObject)morphologizerConfig["labels_pos"]!;
        _labelMorph = morph.Properties().Select(p => (string)p.Value!).ToArray();
        _labelPos = morph.Properties().Select(p => PosNames.TryGetValue((int)pos[p.Name]!, out var name) ? name : "").ToArray();

        // Tokenizer._load_special_cases: every exception; those the affix rules or a space would split also become
        // patterns for the final merge (faster_heuristics).
        var exceptions = (Dictionary<object, object?>)tokenizer["exceptions"]!;
        foreach (var pair in exceptions)
        {
            var pieces = ((List<object?>)pair.Value!).Cast<Dictionary<object, object?>>()
                .Select(attrs => ((string)attrs[65L]!, attrs.TryGetValue(67L, out var norm) ? (string?)norm : null)).ToArray();
            _specials[(string)pair.Key] = pieces;
        }
        foreach (var chunk in _specials.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            if (FindPrefix(chunk) == 0 && FindSuffix(chunk) == 0 && (_infix is null || !_infix.IsMatch(chunk)) && chunk.IndexOf(' ') < 0) continue;
            var orths = TokenizeAffixes(chunk, withSpecialCases: false).Select(t => t.Text).ToArray();
            if (orths.Length == 0) continue;
            if (!_patterns.TryGetValue(orths[0], out var list)) _patterns[orths[0]] = list = new List<string[]>();
            list.Add(orths);
        }
    }

    /// <summary>Loads the pipeline directory (<c>tokenizer</c>, <c>tok2vec/model</c>, <c>morphologizer/model</c>,
    /// <c>morphologizer/cfg</c>).</summary>
    public static SpacyRussianTagger Load(string modelDirectory)
    {
        var tokenizer = (Dictionary<object, object?>)MessagePackReader.Read(File.ReadAllBytes(Path.Combine(modelDirectory, "tokenizer")))!;
        var network = SpacyTok2Vec.Load(Path.Combine(modelDirectory, "tok2vec", "model"), Path.Combine(modelDirectory, "morphologizer", "model"));
        var config = JObject.Parse(File.ReadAllText(Path.Combine(modelDirectory, "morphologizer", "cfg"), Encoding.UTF8));
        return new SpacyRussianTagger(tokenizer, network, config);
    }

    // ---------------------------------------------------------------- Python regular expressions

    /// <summary>A Python <c>re</c> pattern as a .NET one: <c>(?u)</c> dropped (Unicode is the default), and the
    /// <c>\UXXXXXXXX</c> code points of character classes (which .NET cannot hold: it matches UTF-16 units) moved into
    /// alternatives of surrogate pairs.</summary>
    internal static string PythonPattern(string pattern)
    {
        pattern = pattern.Replace("(?u)", "");
        var output = new StringBuilder(pattern.Length);
        for (int i = 0; i < pattern.Length;)
        {
            char c = pattern[i];
            if (c == '\\')
            {
                if (i + 1 < pattern.Length && pattern[i + 1] == 'U')
                {
                    int cp = int.Parse(pattern.Substring(i + 2, 8), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    output.Append(Literal(cp));
                    i += 10;
                }
                else
                {
                    output.Append(pattern, i, Math.Min(2, pattern.Length - i));
                    i += 2;
                }
                continue;
            }
            if (c != '[')
            {
                output.Append(c);
                i++;
                continue;
            }
            i = Class(pattern, i, output);
        }
        return output.ToString();
    }

    private static string Literal(int cp) => cp <= 0xFFFF ? $"\\u{cp:X4}" : string.Concat(char.ConvertFromUtf32(cp).Select(u => $"\\u{(int)u:X4}"));

    // One character class from '[' at start: its BMP part stays a class, its astral ranges become surrogate-pair branches.
    private static int Class(string pattern, int start, StringBuilder output)
    {
        int i = start + 1;
        bool negated = i < pattern.Length && pattern[i] == '^';
        if (negated) i++;
        var bmp = new StringBuilder();
        var astral = new List<(int Low, int High)>();
        bool first = true;
        // A class item: an escape or a character, as text and (when it is one code point) its value.
        (string Text, int? Value) Item(ref int at)
        {
            if (pattern[at] == '\\')
            {
                char e = pattern[at + 1];
                if (e == 'U')
                {
                    int v = int.Parse(pattern.Substring(at + 2, 8), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    at += 10;
                    return (Literal(v), v);
                }
                if (e == 'u')
                {
                    int v = int.Parse(pattern.Substring(at + 2, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    at += 6;
                    return ($"\\u{v:X4}", v);
                }
                if (e == 'x')
                {
                    int v = int.Parse(pattern.Substring(at + 2, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    at += 4;
                    return ($"\\u{v:X4}", v);
                }
                at += 2;
                int? literal = "dDwWsSbB".IndexOf(e) >= 0 ? null : e switch { 'n' => '\n', 't' => '\t', 'r' => '\r', 'f' => '\f', 'v' => '\v', 'a' => '\a', _ => e };
                return (literal is int lv && "dDwWsSbB".IndexOf(e) < 0 ? $"\\u{lv:X4}" : "\\" + e, literal);
            }
            int cp = char.ConvertToUtf32(pattern, at);
            at += char.IsSurrogatePair(pattern, at) ? 2 : 1;
            return (cp <= 0xFFFF ? $"\\u{cp:X4}" : Literal(cp), cp);
        }
        while (i < pattern.Length && (pattern[i] != ']' || first))
        {
            first = false;
            var (text, value) = Item(ref i);
            if (i + 1 < pattern.Length && pattern[i] == '-' && pattern[i + 1] != ']' && value is int low)
            {
                i++;
                var (highText, highValue) = Item(ref i);
                int high = highValue ?? throw new NotSupportedException("A class range ends in a class escape.");
                if (high > 0xFFFF)
                {
                    if (low <= 0xFFFF)
                    {
                        bmp.Append(text).Append('-').Append("\\uFFFF");
                        low = 0x10000;
                    }
                    astral.Add((low, high));
                }
                else
                {
                    bmp.Append(text).Append('-').Append(highText);
                }
                continue;
            }
            if (value is int v && v > 0xFFFF) astral.Add((v, v));
            else bmp.Append(text);
        }
        i++;   // ']'
        if (astral.Count == 0)
        {
            output.Append('[').Append(negated ? "^" : "").Append(bmp).Append(']');
            return i;
        }
        if (negated) throw new NotSupportedException("A negated class with supplementary characters cannot be translated.");
        var branches = new List<string>();
        if (bmp.Length > 0) branches.Add("[" + bmp + "]");
        foreach (var (low, high) in astral) branches.AddRange(SurrogateRanges(low, high));
        output.Append("(?:").Append(string.Join("|", branches)).Append(')');
        return i;
    }

    // The UTF-16 encodings of [low, high] as high-surrogate-prefixed low-surrogate ranges.
    private static IEnumerable<string> SurrogateRanges(int low, int high)
    {
        int hl = 0xD800 + ((low - 0x10000) >> 10), hh = 0xD800 + ((high - 0x10000) >> 10);
        for (int h = hl; h <= hh; h++)
        {
            int from = h == hl ? 0xDC00 + ((low - 0x10000) & 0x3FF) : 0xDC00;
            int to = h == hh ? 0xDC00 + ((high - 0x10000) & 0x3FF) : 0xDFFF;
            yield return $"\\u{h:X4}[\\u{from:X4}-\\u{to:X4}]";
        }
    }

    // ---------------------------------------------------------------- tokenizer

    private int FindPrefix(string s) => _prefix?.Match(s) is { Success: true } m ? m.Length : 0;
    private int FindSuffix(string s) => _suffix?.Match(s) is { Success: true } m ? m.Length : 0;

    private void Special(string chunk, List<Token> tokens)
    {
        foreach (var (orth, norm) in _specials[chunk]) tokens.Add(new Token { Text = orth, Norm = norm });
    }

    // Tokenizer._tokenize_affixes: whitespace chunks, each through the specials or _tokenize.
    private List<Token> TokenizeAffixes(string text, bool withSpecialCases)
    {
        var tokens = new List<Token>();
        if (text.Length == 0) return tokens;
        void Chunk(string span)
        {
            if (withSpecialCases && _specials.ContainsKey(span)) Special(span, tokens);
            else Tokenize(span, tokens, withSpecialCases);
        }
        int start = 0;
        bool inSpace = PythonString.IsSpace(text[0]);
        for (int i = 0; i < text.Length; i++)
        {
            char uc = text[i];
            if (PythonString.IsSpace(uc) == inSpace) continue;
            if (start < i) Chunk(text.Substring(start, i - start));
            if (uc == ' ')
            {
                if (tokens.Count > 0) tokens[tokens.Count - 1].Space = true;
                start = i + 1;
            }
            else
            {
                start = i;
            }
            inSpace = !inSpace;
        }
        if (start < text.Length)
        {
            Chunk(text.Substring(start));
            tokens[tokens.Count - 1].Space = text[text.Length - 1] == ' ' && !inSpace;
        }
        return tokens;
    }

    // Tokenizer._tokenize: _split_affixes, then _attach_tokens.
    private void Tokenize(string text, List<Token> tokens, bool withSpecialCases)
    {
        var prefixes = new List<string>();
        var suffixes = new List<string>();
        string s = text;
        int lastSize = -1;
        while (s.Length > 0 && s.Length != lastSize)
        {
            if (withSpecialCases && _specials.ContainsKey(s)) break;
            lastSize = s.Length;
            int pre = FindPrefix(s);
            string prefix = "", minusPre = "";
            if (pre != 0)
            {
                prefix = s.Substring(0, pre);
                minusPre = s.Substring(pre);
                if (minusPre.Length > 0 && withSpecialCases && _specials.ContainsKey(minusPre))
                {
                    s = minusPre;
                    prefixes.Add(prefix);
                    break;
                }
            }
            int suf = FindSuffix(s.Substring(pre));
            string suffix = "", minusSuf = "";
            if (suf != 0)
            {
                suffix = s.Substring(s.Length - suf);
                minusSuf = s.Substring(0, s.Length - suf);
                if (minusSuf.Length > 0 && withSpecialCases && _specials.ContainsKey(minusSuf))
                {
                    s = minusSuf;
                    suffixes.Add(suffix);
                    break;
                }
            }
            if (pre != 0 && suf != 0 && pre + suf <= s.Length)
            {
                s = s.Substring(pre, s.Length - pre - suf);
                prefixes.Add(prefix);
                suffixes.Add(suffix);
            }
            else if (pre != 0)
            {
                s = minusPre;
                prefixes.Add(prefix);
            }
            else if (suf != 0)
            {
                s = minusSuf;
                suffixes.Add(suffix);
            }
        }

        foreach (var p in prefixes) tokens.Add(new Token { Text = p });
        if (s.Length > 0)
        {
            if (withSpecialCases && _specials.ContainsKey(s))
            {
                Special(s, tokens);
            }
            else if (_url?.Match(s) is { Success: true, Index: 0 })
            {
                tokens.Add(new Token { Text = s });
            }
            else
            {
                var matches = _infix is null ? new List<Match>() : _infix.Matches(s).Cast<Match>().ToList();
                if (matches.Count == 0)
                {
                    tokens.Add(new Token { Text = s });
                }
                else
                {
                    int start = 0;
                    foreach (var match in matches)
                    {
                        int infixStart = match.Index, infixEnd = match.Index + match.Length;
                        if (infixStart == 0) continue;
                        if (infixStart != start) tokens.Add(new Token { Text = s.Substring(start, infixStart - start) });
                        if (infixStart != infixEnd) tokens.Add(new Token { Text = s.Substring(infixStart, infixEnd - infixStart) });
                        start = infixEnd;
                    }
                    if (start < s.Length) tokens.Add(new Token { Text = s.Substring(start) });
                }
            }
        }
        for (int i = suffixes.Count - 1; i >= 0; i--) tokens.Add(new Token { Text = suffixes[i] });
    }

    /// <summary><c>Tokenizer.__call__</c>: the affix tokenization, then the special cases that span several of its
    /// tokens (longest first, earlier on ties, non-overlapping) merged.</summary>
    internal List<Token> Tokenize(string text)
    {
        var tokens = TokenizeAffixes(text, withSpecialCases: true);
        var spans = new List<(int Start, int End)>();
        for (int i = 0; i < tokens.Count; i++)
            if (_patterns.TryGetValue(tokens[i].Text, out var candidates))
                foreach (var pattern in candidates)
                    if (i + pattern.Length <= tokens.Count && pattern.Select((t, k) => t == tokens[i + k].Text).All(x => x))
                        spans.Add((i, i + pattern.Length));
        if (spans.Count == 0) return tokens;
        // _filter_special_spans: by length then start (later start first on equal length), taken from the end.
        spans.Sort((a, b) => a.End - a.Start != b.End - b.Start ? (a.End - a.Start).CompareTo(b.End - b.Start) : b.Start.CompareTo(a.Start));
        var seen = new HashSet<int>();
        var filtered = new List<(int Start, int End)>();
        for (int k = spans.Count - 1; k >= 0; k--)
        {
            var span = spans[k];
            if (!seen.Contains(span.Start) && !seen.Contains(span.End - 1)) filtered.Add(span);
            for (int t = span.Start; t < span.End; t++) seen.Add(t);
        }
        var byStart = filtered.GroupBy(s => s.Start).ToDictionary(g => g.Key, g => g.Last());
        var output = new List<Token>();
        for (int i = 0; i < tokens.Count;)
        {
            if (!byStart.TryGetValue(i, out var span))
            {
                output.Add(tokens[i++]);
                continue;
            }
            var text_ = new StringBuilder();
            for (int t = span.Start; t < span.End; t++) text_.Append(t < span.End - 1 ? tokens[t].TextWithWhitespace : tokens[t].Text);
            if (_specials.ContainsKey(text_.ToString()))
            {
                int before = output.Count;
                Special(text_.ToString(), output);
                output[output.Count - 1].Space = tokens[span.End - 1].Space;
            }
            else
            {
                for (int t = span.Start; t < span.End; t++) output.Add(tokens[t]);
            }
            i = span.End;
        }
        return output;
    }

    // ---------------------------------------------------------------- tagging

    /// <summary>The text's tokens with their POS and morphology.</summary>
    public List<Token> Tag(string text)
    {
        var tokens = Tokenize(text);
        if (tokens.Count == 0) return tokens;
        var labels = _network.Predict(tokens.Select(Features).ToArray());
        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            token.Pos = _labelPos[labels[i]];
            token.Morph = _labelMorph[labels[i]];
            // The attribute ruler: whitespace is SPACE without features; SPACE on anything else is X.
            bool isSpace = PythonString.IsSpace(token.Text);
            if (isSpace || token.Pos == "SPACE")
            {
                token.Pos = isSpace ? "SPACE" : "X";
                token.Morph = "";
            }
        }
        return tokens;
    }

    /// <summary>The six attributes tok2vec reads (NORM, PREFIX, SUFFIX, SHAPE as string hashes; SPACY, IS_SPACE as
    /// 0 or 1).</summary>
    internal static ulong[] Features(Token token)
    {
        var glyphs = ChineseText.Codepoints(token.Text);
        string norm = token.Norm ?? (BaseNorms.TryGetValue(token.Text, out var shared) ? shared : PythonString.Lower(token.Text));
        string prefix = glyphs.Count > 0 ? glyphs[0] : "";
        string suffix = string.Concat(glyphs.Skip(Math.Max(0, glyphs.Count - 3)));
        return new[]
        {
            StringId(norm), StringId(prefix), StringId(suffix), StringId(Shape(glyphs)),
            token.Space ? 1UL : 0UL, PythonString.IsSpace(token.Text) ? 1UL : 0UL,
        };
    }

    // spacy.lang.lex_attrs.word_shape.
    private static string Shape(List<string> glyphs)
    {
        if (glyphs.Count >= 100) return "LONG";
        var shape = new StringBuilder();
        string last = "";
        int seq = 0;
        foreach (var g in glyphs)
        {
            string c = PythonString.IsAlpha(g) ? (PythonString.IsUpper(g) ? "X" : "x") : PythonString.IsDigit(g) ? "d" : g;
            if (c == last) seq++;
            else
            {
                seq = 0;
                last = c;
            }
            if (seq < 4) shape.Append(c);
        }
        return shape.ToString();
    }

    /// <summary>The id spaCy's <c>StringStore</c> gives a string: 0 for the empty string, a predefined symbol's index,
    /// else its hash.</summary>
    internal static ulong StringId(string text) =>
        text.Length == 0 ? 0UL : SpacySymbols.TryGetId(text, out int id) ? (ulong)id : HashString(text);

    /// <summary>spaCy's <c>hash_string</c>: MurmurHash64A of the UTF-8 bytes with seed 1.</summary>
    internal static ulong HashString(string text)
    {
        var data = Encoding.UTF8.GetBytes(text);
        const ulong m = 0xc6a4a7935bd1e995UL;
        const int r = 47;
        ulong h = 1UL ^ ((ulong)data.Length * m);
        int blocks = data.Length / 8;
        for (int i = 0; i < blocks; i++)
        {
            ulong k = BitConverter.ToUInt64(data, i * 8);
            k *= m;
            k ^= k >> r;
            k *= m;
            h ^= k;
            h *= m;
        }
        int tail = blocks * 8;
        switch (data.Length & 7)
        {
            case 7: h ^= (ulong)data[tail + 6] << 48; goto case 6;
            case 6: h ^= (ulong)data[tail + 5] << 40; goto case 5;
            case 5: h ^= (ulong)data[tail + 4] << 32; goto case 4;
            case 4: h ^= (ulong)data[tail + 3] << 24; goto case 3;
            case 3: h ^= (ulong)data[tail + 2] << 16; goto case 2;
            case 2: h ^= (ulong)data[tail + 1] << 8; goto case 1;
            case 1:
                h ^= data[tail];
                h *= m;
                break;
        }
        h ^= h >> r;
        h *= m;
        h ^= h >> r;
        return h;
    }
}
