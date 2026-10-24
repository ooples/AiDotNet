using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// NLTK's English averaged-perceptron part-of-speech tagger (<c>nltk.tag.perceptron.PerceptronTagger</c> with the
/// <c>averaged_perceptron_tagger_eng</c> model), as <c>nltk.pos_tag</c> runs it.
/// </summary>
internal sealed class AveragedPerceptronTagger
{
    private const string ResourceName = "AiDotNet.TextToSpeech.FrontEnd.perceptron_tagger_eng.json.gz";
    private static readonly Lazy<AveragedPerceptronTagger> Shared = new(() => new AveragedPerceptronTagger());
    private readonly Dictionary<string, List<(string Label, double Weight)>> _weights = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _tagdict = new(StringComparer.Ordinal);
    private readonly List<string> _classes = new();

    public static AveragedPerceptronTagger English => Shared.Value;

    private AveragedPerceptronTagger()
    {
        using var stream = typeof(AveragedPerceptronTagger).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidDataException($"The embedded resource {ResourceName} is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var document = JsonDocument.Parse(gzip);
        foreach (var feature in document.RootElement.GetProperty("weights").EnumerateObject())
            _weights[feature.Name] = feature.Value.EnumerateObject().Select(p => (p.Name, p.Value.GetDouble())).ToList();
        foreach (var entry in document.RootElement.GetProperty("tagdict").EnumerateObject()) _tagdict[entry.Name] = entry.Value.GetString()!;
        foreach (var label in document.RootElement.GetProperty("classes").EnumerateArray()) _classes.Add(label.GetString()!);
    }

    private static readonly string[] Start = { "-START-", "-START2-" };
    private static readonly string[] End = { "-END-", "-END2-" };

    private static string Normalize(string word)
    {
        if (word.Contains('-') && word[0] != '-') return "!HYPHEN";
        if (word.Length == 4 && word.All(char.IsDigit)) return "!YEAR";
        if (word.Length > 0 && char.IsDigit(word[0])) return "!DIGITS";
        return word.ToLowerInvariant();
    }

    private static string Last3(string s) => s.Length <= 3 ? s : s.Substring(s.Length - 3);

    /// <summary>The tag of each token (<c>PerceptronTagger.tag</c>).</summary>
    public List<string> Tag(IReadOnlyList<string> tokens)
    {
        string prev = Start[0], prev2 = Start[1];
        var context = Start.Concat(tokens.Select(Normalize)).Concat(End).ToList();
        var output = new List<string>();
        for (int i = 0; i < tokens.Count; i++)
        {
            string word = tokens[i];
            if (!_tagdict.TryGetValue(word, out var tag))
            {
                int c = i + Start.Length;
                // Features in _get_features' order; each counts once.
                var features = new List<string>
                {
                    "bias",
                    "i suffix " + Last3(word),
                    "i pref1 " + (word.Length > 0 ? word.Substring(0, 1) : ""),
                    "i-1 tag " + prev,
                    "i-2 tag " + prev2,
                    $"i tag+i-2 tag {prev} {prev2}",
                    "i word " + context[c],
                    $"i-1 tag+i word {prev} {context[c]}",
                    "i-1 word " + context[c - 1],
                    "i-1 suffix " + Last3(context[c - 1]),
                    "i-2 word " + context[c - 2],
                    "i+1 word " + context[c + 1],
                    "i+1 suffix " + Last3(context[c + 1]),
                    "i+2 word " + context[c + 2],
                };
                var counts = new Dictionary<string, int>(StringComparer.Ordinal);
                var order = new List<string>();
                foreach (var f in features)
                {
                    if (!counts.ContainsKey(f)) { counts[f] = 0; order.Add(f); }
                    counts[f]++;
                }
                var scores = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (var f in order)
                {
                    if (!_weights.TryGetValue(f, out var weights)) continue;
                    foreach (var (label, weight) in weights)
                        scores[label] = (scores.TryGetValue(label, out var s) ? s : 0.0) + counts[f] * weight;
                }
                // max over the classes by (score, label).
                tag = _classes.OrderByDescending(l => scores.TryGetValue(l, out var s) ? s : 0.0).ThenByDescending(l => l, StringComparer.Ordinal).First();
            }
            output.Add(tag);
            prev2 = prev;
            prev = tag;
        }
        return output;
    }
}

/// <summary>
/// g2p_en 2.1.0 (Kyubyong Park; Apache-2.0), the English grapheme-to-phoneme converter MeloTTS falls back on: numbers
/// spelled out, accents stripped, lower case and only <c>a–z ' . , ? ! -</c> kept, NLTK's tokenizer and part-of-speech
/// tagger, then per word its homograph pronunciation for the tagged part of speech, else its CMU dictionary (0.7b)
/// pronunciation, else the prediction of g2p_en's GRU encoder–decoder.
/// </summary>
/// <remarks>MeloTTS calls it on one word at a time (a BERT word without the word pieces' "##"), where NLTK's word
/// tokenizer leaves the word whole; <see cref="Word"/> is that use.</remarks>
internal sealed class G2pEn
{
    private static readonly Lazy<G2pEn> Shared = new(() => new G2pEn());
    private readonly Dictionary<string, string[]> _cmu = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (string[] First, string[] Second, string Pos)> _homographs = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float[]> _w = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int[]> _shape = new(StringComparer.Ordinal);

    private static readonly string[] Graphemes = new[] { "<pad>", "<unk>", "</s>" }.Concat("abcdefghijklmnopqrstuvwxyz".Select(c => c.ToString())).ToArray();
    private static readonly string[] Phonemes =
    {
        "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2",
        "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1",
        "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
        "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    };

    public static G2pEn Default => Shared.Value;

    private static Stream Resource(string name) =>
        new GZipStream(typeof(G2pEn).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd." + name)
            ?? throw new InvalidDataException($"The embedded resource {name} is missing."), CompressionMode.Decompress);

    private G2pEn()
    {
        using (var reader = new StreamReader(Resource("g2pen_cmudict.tsv.gz"), Encoding.UTF8))
        {
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                int tab = line.IndexOf('\t');
                if (tab > 0) _cmu[line.Substring(0, tab)] = line.Substring(tab + 1).Split(' ');
            }
        }
        using (var reader = new StreamReader(Resource("g2pen_homographs.tsv.gz"), Encoding.UTF8))
        {
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (line.StartsWith("#", StringComparison.Ordinal)) continue;
                var parts = line.Trim().Split('|');
                if (parts.Length != 4) continue;
                _homographs[parts[0].ToLowerInvariant()] = (parts[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries),
                    parts[2].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries), parts[3]);
            }
        }
        using (var stream = Resource("g2pen_model.f32.gz"))
        using (var reader = new BinaryReader(stream))
        {
            int count = reader.ReadInt32();
            for (int i = 0; i < count; i++)
            {
                string name = Encoding.ASCII.GetString(reader.ReadBytes(reader.ReadInt32()));
                var shape = new int[reader.ReadInt32()];
                for (int d = 0; d < shape.Length; d++) shape[d] = reader.ReadInt32();
                var values = new float[shape.Aggregate(1, (a, b) => a * b)];
                for (int k = 0; k < values.Length; k++) values[k] = reader.ReadSingle();
                _w[name] = values;
                _shape[name] = shape;
            }
        }
    }

    private static readonly Regex Keep = new(@"[^ a-z'.,?!\-]", RegexOptions.CultureInvariant);
    private static readonly Regex HasLetter = new("[a-z]", RegexOptions.CultureInvariant);

    /// <summary>
    /// <c>G2p()(word)</c> for text that NLTK's word tokenizer keeps as one token or none (MeloTTS's case): the
    /// pronunciation's phones (ARPAbet with stress), or the text itself when it has no letter.
    /// </summary>
    public List<string> Word(string text)
    {
        text = G2pEnNumbers.Normalize(text);
        text = new string(text.Normalize(NormalizationForm.FormD)
            .Where(c => System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c) != System.Globalization.UnicodeCategory.NonSpacingMark).ToArray());
        text = Keep.Replace(text.ToLowerInvariant(), "");
        text = text.Replace("i.e.", "that is").Replace("e.g.", "for example");
        var words = NltkWordTokens(text);
        var tags = words.Count > 0 ? AveragedPerceptronTagger.English.Tag(words) : new List<string>();
        var prons = new List<string>();
        for (int i = 0; i < words.Count; i++)
        {
            string word = words[i];
            IEnumerable<string> pron;
            if (!HasLetter.IsMatch(word)) pron = new[] { word };
            else if (_homographs.TryGetValue(word, out var h)) pron = tags[i].StartsWith(h.Pos, StringComparison.Ordinal) ? h.First : h.Second;
            else if (_cmu.TryGetValue(word, out var phones)) pron = phones;
            else pron = Predict(word);
            prons.AddRange(pron);
            prons.Add(" ");
        }
        if (prons.Count > 0) prons.RemoveAt(prons.Count - 1);
        return prons;
    }

    // nltk.word_tokenize for one word at a time (MeloTTS's BERT tokens, EmotiVoice's words): no spaces inside a word, so
    // splitting at spaces and splitting off a final period, comma, question or exclamation mark (the Treebank tokenizer's
    // rules for a single token) covers the punctuation; then NLTKWordTokenizer's quote and contraction rules: a leading
    // apostrophe before a word ("'tis" excepted), a final 's, 'm, 'd, ', 'll, 're, 've or n't, and MacIntyre's
    // contractions (can|not, gon|na, ...).
    private static readonly Regex StartingQuote = new(@"(?<!\w)(')(?!(?:re|ve|ll|m|t|s|d|n)\b)(?=\w)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    private static readonly Regex[] EndingQuotes =
    {
        new(@"([^' ])('[sS]|'[mM]|'[dD]|') ", RegexOptions.CultureInvariant),
        new(@"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", RegexOptions.CultureInvariant),
    };
    private static readonly Regex[] Contractions =
    {
        new(@"\b(can)(not)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(d)('ye)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(gim)(me)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(gon)(na)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(got)(ta)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(lem)(me)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(more)('n)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@"\b(wan)(na)(?=\s)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@" ('t)(is)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
        new(@" ('t)(was)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
    };

    private static List<string> NltkWordTokens(string text)
    {
        var tokens = new List<string>();
        foreach (var raw in text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string piece = StartingQuote.Replace(raw, "$1 ");
            string? mark = null;
            if (piece.Length > 1 && (piece.EndsWith(".", StringComparison.Ordinal) || piece.EndsWith(",", StringComparison.Ordinal)
                || piece.EndsWith("?", StringComparison.Ordinal) || piece.EndsWith("!", StringComparison.Ordinal)))
            {
                mark = piece.Substring(piece.Length - 1);
                piece = piece.Substring(0, piece.Length - 1);
            }
            piece = " " + piece + " ";
            foreach (var rule in EndingQuotes) piece = rule.Replace(piece, "$1 $2 ");
            foreach (var rule in Contractions) piece = rule.Replace(piece, " $1 $2 ");
            tokens.AddRange(piece.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));
            if (mark is not null) tokens.Add(mark);
        }
        return tokens;
    }

    private float[] W(string name) => _w[name];

    // One GRU step (PyTorch gate order r, z, n), as g2p_en's numpy grucell, in float32.
    private float[] Cell(float[] x, float[] h, string prefix)
    {
        var wih = W(prefix + "_w_ih"); var whh = W(prefix + "_w_hh"); var bih = W(prefix + "_b_ih"); var bhh = W(prefix + "_b_hh");
        int hidden = h.Length, input = x.Length;
        var gi = new float[3 * hidden];
        var gh = new float[3 * hidden];
        for (int r = 0; r < 3 * hidden; r++)
        {
            float si = 0, sh = 0;
            for (int k = 0; k < input; k++) si += x[k] * wih[r * input + k];
            for (int k = 0; k < hidden; k++) sh += h[k] * whh[r * hidden + k];
            gi[r] = si + bih[r];
            gh[r] = sh + bhh[r];
        }
        var next = new float[hidden];
        for (int j = 0; j < hidden; j++)
        {
            float rr = Sigmoid(gi[j] + gh[j]);
            float z = Sigmoid(gi[hidden + j] + gh[hidden + j]);
            float n = (float)Math.Tanh(gi[2 * hidden + j] + rr * gh[2 * hidden + j]);
            next[j] = (1 - z) * n + z * h[j];
        }
        return next;
    }

    private static float Sigmoid(float x) => (float)(1 / (1 + Math.Exp(-x)));

    private float[] Row(string table, int index)
    {
        int width = _shape[table][1];
        var row = new float[width];
        Array.Copy(W(table), index * width, row, 0, width);
        return row;
    }

    /// <summary>g2p_en's neural prediction for an out-of-vocabulary word: a greedy decode of at most 20 phones.</summary>
    public List<string> Predict(string word)
    {
        int hidden = _shape["enc_w_hh"][1];
        var h = new float[hidden];
        foreach (var c in word.Select(ch => ch.ToString()).Append("</s>"))
        {
            int index = Array.IndexOf(Graphemes, c);
            h = Cell(Row("enc_emb", index < 0 ? 1 : index), h, "enc");
        }
        var dec = Row("dec_emb", 2);
        var fcw = W("fc_w"); var fcb = W("fc_b");
        int outputs = fcb.Length;
        var predictions = new List<string>();
        for (int step = 0; step < 20; step++)
        {
            h = Cell(dec, h, "dec");
            int best = 0;
            float bestValue = float.NegativeInfinity;
            for (int o = 0; o < outputs; o++)
            {
                float v = fcb[o];
                for (int k = 0; k < hidden; k++) v += h[k] * fcw[o * hidden + k];
                if (v > bestValue) { bestValue = v; best = o; }
            }
            if (best == 3) break;
            predictions.Add(best < Phonemes.Length ? Phonemes[best] : "<unk>");
            dec = Row("dec_emb", best);
        }
        return predictions;
    }
}

/// <summary>g2p_en's <c>expand.normalize_numbers</c> (after keithito/tacotron): numbers, currency and ordinals in words.</summary>
internal static class G2pEnNumbers
{
    private static readonly Regex CommaNumber = new(@"([0-9][0-9\,]+[0-9])", RegexOptions.CultureInvariant);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Pounds = new(@"£([0-9\,]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Dollars = new(@"\$([0-9\.\,]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Ordinal = new(@"[0-9]+(st|nd|rd|th)", RegexOptions.CultureInvariant);
    private static readonly Regex Number = new(@"[0-9]+", RegexOptions.CultureInvariant);

    public static string Normalize(string text)
    {
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Pounds.Replace(text, "$1 pounds");
        text = Dollars.Replace(text, m =>
        {
            var parts = m.Groups[1].Value.Split('.');
            if (parts.Length > 2) return m.Groups[1].Value + " dollars";
            var dollars = parts[0].Length > 0 ? System.Numerics.BigInteger.Parse(parts[0]) : 0;
            var cents = parts.Length > 1 && parts[1].Length > 0 ? System.Numerics.BigInteger.Parse(parts[1]) : 0;
            string du = dollars == 1 ? "dollar" : "dollars", cu = cents == 1 ? "cent" : "cents";
            if (!dollars.IsZero && !cents.IsZero) return $"{dollars} {du}, {cents} {cu}";
            if (!dollars.IsZero) return $"{dollars} {du}";
            if (!cents.IsZero) return $"{cents} {cu}";
            return "zero dollars";
        });
        text = DecimalNumber.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = Ordinal.Replace(text, m => InflectNumberWords.NumberToWords(m.Value));
        return Number.Replace(text, m => ExpandNumber(m.Value, zeroOhAboveThousand: true));
    }

    /// <summary>keithito's <c>_expand_number</c>: years 1001–2999 read in pairs, other numbers with no "and".</summary>
    public static string ExpandNumber(string digits, bool zeroOhAboveThousand)
    {
        var num = System.Numerics.BigInteger.Parse(digits);
        if (num > 1000 && num < 3000)
        {
            if (num == 2000) return "two thousand";
            if (num > 2000 && num < 2010) return "two thousand " + InflectNumberWords.NumberToWords((num % 100).ToString());
            if (num % 100 == 0) return InflectNumberWords.NumberToWords((num / 100).ToString()) + " hundred";
            return InflectNumberWords.NumberToWords(num.ToString(), group: 2, andword: "", zero: "oh").Replace(", ", " ");
        }
        return InflectNumberWords.NumberToWords(num.ToString(), andword: "");
    }
}
