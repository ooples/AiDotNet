using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// MaskGCT's Mandarin front end (Amphion <c>models/tts/maskgct/g2p/g2p/mandarin.py</c>): numbers through cn2an,
/// punctuation normalized and other characters dropped, jieba's words (with 一, 不 and 儿 merged), each word read from the
/// release's lexicon or character by character — a polyphonic character by the polyphone BERT — with the third-tone, 不,
/// 一 and 儿 sandhi, then bopomofo to IPA with tone arrows.
/// </summary>
internal sealed class MaskGctMandarinFrontEnd
{
    private sealed class Data
    {
        public Dictionary<string, string> Lexicon { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, string> PinyinToBopomofo { get; } = new(StringComparer.Ordinal);
        public HashSet<string> Polyphonic { get; } = new(StringComparer.Ordinal);
        public Dictionary<int, string> PolyphoneLabels { get; } = new();
        public Dictionary<string, int> BertVocabulary { get; } = new(StringComparer.Ordinal);
        public List<(Regex Pattern, string Replacement)> LatinToBopomofo { get; } = new();
        public List<(Regex Pattern, string Replacement)> BopomofoToIpa { get; } = new();
        public List<(int From, int To)> NumericRanges { get; } = new();
    }

    private static readonly Lazy<Data> Resources = new(() =>
    {
        using var stream = typeof(MaskGctMandarinFrontEnd).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.maskgct_mandarin.json.gz")
            ?? throw new InvalidDataException("The embedded MaskGCT Mandarin data is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var json = JObject.Parse(reader.ReadToEnd());
        var data = new Data();
        foreach (var pair in (JArray)json["lexicon"]!) data.Lexicon[(string)pair[0]!] = (string)pair[1]!;
        foreach (var pair in (JArray)json["pinyin_to_bopomofo"]!) data.PinyinToBopomofo[(string)pair[0]!] = (string)pair[1]!;
        foreach (var c in (JArray)json["polychar"]!) data.Polyphonic.Add((string)c!);
        foreach (var p in ((JObject)json["polydict"]!).Properties()) data.PolyphoneLabels[int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture)] = (string)p.Value!;
        var vocab = (JArray)json["bert_vocab"]!;
        for (int i = 0; i < vocab.Count; i++) data.BertVocabulary[(string)vocab[i]!] = i;
        foreach (var pair in (JArray)json["latin_to_bopomofo"]!)
            data.LatinToBopomofo.Add((new Regex((string)pair[0]!, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant), (string)pair[1]!));
        foreach (var range in (JArray)json["numeric_ranges"]!) data.NumericRanges.Add(((int)range[0]!, (int)range[1]!));
        foreach (var pair in (JArray)json["bopomofo_to_ipa"]!)
            data.BopomofoToIpa.Add((new Regex((string)pair[0]!, RegexOptions.CultureInvariant), (string)pair[1]!));
        return data;
    }, isThreadSafe: true);

    private static readonly Dictionary<string, string> Tones = new(StringComparer.Ordinal)
    {
        ["0"] = "˙", ["5"] = "˙", ["1"] = "", ["2"] = "ˊ", ["3"] = "ˇ", ["4"] = "ˋ",
    };

    private static readonly HashSet<string> MustNotEr = new(StringComparer.Ordinal) { "女儿", "老儿", "男儿", "少儿", "小儿" };

    private readonly Func<string, int, string> _polyphone;

    /// <param name="polyphone">The pinyin with tone digit ("le5") of the character at an index of a text, as the release's
    /// polyphone BERT predicts it (<see cref="MaskGctPolyphoneModel{T}.Predict"/>).</param>
    public MaskGctMandarinFrontEnd(Func<string, int, string> polyphone) => _polyphone = polyphone ?? throw new ArgumentNullException(nameof(polyphone));

    /// <summary>The polyphone model's label for a class index (<c>polydict.json</c>: "字{pinyin}").</summary>
    internal static string PolyphoneLabel(int index) => Resources.Value.PolyphoneLabels[index];

    /// <summary>The polyphone BERT's token ids of a text: [CLS], then each character's id ([UNK] when absent).</summary>
    internal static int[] BertIds(string text)
    {
        var vocabulary = Resources.Value.BertVocabulary;
        int unknown = vocabulary["[UNK]"];
        return new[] { vocabulary["[CLS]"] }.Concat(ChineseText.Codepoints(text).Select(c => vocabulary.TryGetValue(c, out int id) ? id : unknown)).ToArray();
    }

    // Python 3.9's str.isnumeric for one code point (the release's Unicode tables, exported with the data).
    private static bool IsNumeric(string codepoint)
    {
        if (codepoint.Length == 0) return false;
        int code = char.ConvertToUtf32(codepoint, 0);
        var ranges = Resources.Value.NumericRanges;
        int lo = 0, hi = ranges.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (code < ranges[mid].From) hi = mid - 1;
            else if (code > ranges[mid].To) lo = mid + 1;
            else return true;
        }
        return false;
    }

    private static readonly HashSet<string> ChineseDigits = new(StringComparer.Ordinal) { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十" };

    // ---------------------------------------------------------------- word merges

    private static List<string> MergeYi(List<string> seg)
    {
        var merged = new List<string>();
        for (int i = 0; i < seg.Count; i++)
        {
            string word = seg[i];
            if (i - 1 >= 0 && word == "一" && i + 1 < seg.Count && seg[i - 1] == seg[i + 1])
            {
                if (i - 1 < merged.Count) merged[i - 1] = merged[i - 1] + "一" + merged[i - 1];
                else
                {
                    merged.Add(word);
                    merged.Add(seg[i + 1]);
                }
            }
            else if (!(i - 2 >= 0 && seg[i - 1] == "一" && seg[i - 2] == word))
            {
                merged.Add(word);
            }
        }
        var joined = new List<string>();
        foreach (var word in merged)
        {
            if (joined.Count > 0 && joined[joined.Count - 1] == "一") joined[joined.Count - 1] += word;
            else joined.Add(word);
        }
        return joined;
    }

    private static List<string> MergeBu(List<string> seg)
    {
        var merged = new List<string>();
        string last = "";
        foreach (var w in seg)
        {
            string word = w;
            if (word != "不")
            {
                if (last == "不") word = last + word;
                merged.Add(word);
            }
            last = word;
        }
        return merged;
    }

    private static List<string> MergeEr(List<string> seg)
    {
        var merged = new List<string>();
        for (int i = 0; i < seg.Count; i++)
        {
            if (i - 1 >= 0 && seg[i] == "儿") merged[merged.Count - 1] += seg[i];
            else merged.Add(seg[i]);
        }
        return merged;
    }

    // ---------------------------------------------------------------- tone sandhi

    private static string Last(string s) => s.Length == 0 ? "" : s.Substring(s.Length - 1);

    private static string ChangeTone(string bopomofo, string tone) =>
        "˙ˊˇˋ".Contains(Last(bopomofo)) && bopomofo.Length > 0 ? bopomofo.Substring(0, bopomofo.Length - 1) + tone : bopomofo + tone;

    private static void ErSandhi(List<string> word, List<string> b)
    {
        if (word.Count > 1 && word[word.Count - 1] == "儿" && !MustNotEr.Contains(string.Concat(word))) b[b.Count - 1] = ChangeTone(b[b.Count - 1], "˙");
    }

    private static void BuSandhi(List<string> word, List<string> b)
    {
        string text = string.Concat(word);
        if (word.Distinct().Count() == 1 && word[0] == "不") return;
        if (text == "不字") return;
        if (word.Count == 3 && word[1] == "不" && b[1].Substring(0, b[1].Length - 1) == "ㄅㄨ")
        {
            b[1] = b[1].Substring(0, b[1].Length - 1) + "˙";
            return;
        }
        for (int i = 0; i < word.Count; i++)
            if (i + 1 < b.Count && word[i] == "不" && i + 1 < word.Count && b[i + 1].Length > 0 && Last(b[i + 1]) == "ˋ")
                b[i] = b[i].Substring(0, b[i].Length - 1) + "ˊ";
    }

    private const string YiPunctuation = "：，；。？！“”‘’':,;.?!()（）{}【】[]-~`、 ";

    private static void YiSandhi(List<string> word, List<string> b)
    {
        string text = string.Concat(word);
        if (text.Contains("一") && word.Where(c => c != "一").Any(IsNumeric))
        {
            for (int i = 0; i < word.Count; i++)
            {
                if (i == 0 && word[0] == "一" && word.Count > 1 && !ChineseDigits.Contains(word[1]))
                {
                    b[0] = b[0].Length > 0 && (Last(b[1]) == "ˋ" || Last(b[1]) == "˙") ? ChangeTone(b[0], "ˊ") : ChangeTone(b[0], "ˋ");
                }
                else if (word[i] == "一")
                {
                    b[i] = ChangeTone(b[i], "");
                }
            }
            return;
        }
        if (word.Count == 3 && word[1] == "一" && word[0] == word[word.Count - 1])
        {
            b[1] = ChangeTone(b[1], "˙");
        }
        else if (text.StartsWith("第一", StringComparison.Ordinal))
        {
            b[1] = ChangeTone(b[1], "");
        }
        else if (text.StartsWith("一月", StringComparison.Ordinal) || text.StartsWith("一日", StringComparison.Ordinal) || text.StartsWith("一号", StringComparison.Ordinal))
        {
            b[0] = ChangeTone(b[0], "");
        }
        else
        {
            for (int i = 0; i < word.Count; i++)
            {
                if (word[i] != "一" || i + 1 >= word.Count) continue;
                if (b.Count > i + 1 && b[i + 1].Length > 0 && Last(b[i + 1]) == "ˋ") b[i] = ChangeTone(b[i], "ˊ");
                else if (!YiPunctuation.Contains(word[i + 1])) b[i] = ChangeTone(b[i], "ˋ");
            }
        }
    }

    // ---------------------------------------------------------------- readings

    private static readonly Regex Han = new("[一-鿿]", RegexOptions.CultureInvariant);
    private static readonly Regex BareInitial = new("([ㄅ-ㄩ])$", RegexOptions.CultureInvariant);

    private static string Syllable(string pinyin)
    {
        var data = Resources.Value;
        return data.PinyinToBopomofo[pinyin.Substring(0, pinyin.Length - 1)] + Tones[pinyin.Substring(pinyin.Length - 1)];
    }

    /// <summary><c>chinese_to_bopomofo</c>.</summary>
    private string ToBopomofo(string text)
    {
        var data = Resources.Value;
        var words = MergeEr(MergeBu(MergeYi(Jieba.Default.Cut(text))));
        var output = new StringBuilder();
        int charIndex = 0;
        foreach (var word in words)
        {
            var chars = ChineseText.Codepoints(word);
            var b = new List<string>();
            if (data.Lexicon.TryGetValue(word, out var pinyin) && !data.Polyphonic.Contains(word))
            {
                foreach (var py in pinyin.Split(' '))
                {
                    if (py.Length > 0 && data.PinyinToBopomofo.ContainsKey(py.Substring(0, py.Length - 1)) && Tones.ContainsKey(py.Substring(py.Length - 1)))
                        b.Add(Syllable(py));
                    else
                        b.AddRange(PypinyinReader.Bopomofo.Read(word));
                }
            }
            else
            {
                for (int i = 0; i < chars.Count; i++)
                {
                    string c = chars[i];
                    if (data.Polyphonic.Contains(c))
                    {
                        b.Add(Syllable(_polyphone(text, charIndex + i)));
                    }
                    else if (data.Lexicon.TryGetValue(c, out var py))
                    {
                        b.Add(Syllable(py));
                    }
                    else
                    {
                        b.Add(c);
                    }
                }
            }
            charIndex += chars.Count;
            if (chars.Count == 3 && Last(b[0]) == "ˇ" && Last(b[1]) == "ˇ" && Last(b[b.Count - 1]) == "ˇ")
            {
                b[0] += "ˊ";
                b[1] += "ˊ";
            }
            if (chars.Count == 2 && Last(b[0]) == "ˇ" && Last(b[b.Count - 1]) == "ˇ") b[0] = b[0].Substring(0, b[0].Length - 1) + "ˊ";
            BuSandhi(chars, b);
            YiSandhi(chars, b);
            ErSandhi(chars, b);
            if (!Han.IsMatch(word))
            {
                output.Append('|').Append(word);
                continue;
            }
            for (int i = 0; i < b.Count; i++) b[i] = BareInitial.Replace(b[i], "$1ˉ");
            if (output.Length > 0) output.Append('|');
            output.Append(string.Join("|", b));
        }
        return output.ToString();
    }

    private static readonly (string From, string To)[] Punctuation =
    {
        ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("；", ";"), ("：", ":"), ("、", ","), ("‘", "'"), ("’", "'"), ("⋯", "…"),
        ("···", "…"), ("・・・", "…"), ("...", "…"),
    };

    private const string Space = @"\s\x1c-\x1f";
    private static readonly Regex Spaces = new(@"[" + Space + @"]+", RegexOptions.CultureInvariant);
    private static readonly Regex NotKept = new(@"[^一-鿿" + Space + @"_,\.\?!;:'…]", RegexOptions.CultureInvariant);
    private static readonly Regex AroundMark = new(@"[" + Space + @"]*([,\.\?!;:'…])[" + Space + @"]*", RegexOptions.CultureInvariant);
    private static readonly Regex RetroflexEnd = new("([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", RegexOptions.CultureInvariant);
    private static readonly Regex SibilantEnd = new("([s][⁼ʰ]?)([→↓↑ ]+|$)", RegexOptions.CultureInvariant);
    private static readonly Regex Marks = new(@"([,\.\?!;:'…])", RegexOptions.CultureInvariant);
    private static readonly Regex Bars = new(@"\|+", RegexOptions.CultureInvariant);

    // Python's \w over str: letters and numbers of every script, and '_'.
    private static bool IsPythonWord(string s, int i)
    {
        var category = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(s, i);
        return s[i] == '_' || category <= System.Globalization.UnicodeCategory.OtherLetter
            || category is System.Globalization.UnicodeCategory.DecimalDigitNumber or System.Globalization.UnicodeCategory.LetterNumber
                or System.Globalization.UnicodeCategory.OtherNumber;
    }

    // re.sub(r"^\||[^\w\s_,\.\?!;:\'…\|→↓↑⁼ʰ`]", "", text): a leading "|" and every character outside the set.
    private static string KeepIpaCharacters(string text)
    {
        if (text.StartsWith("|", StringComparison.Ordinal)) text = text.Substring(1);
        var sb = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i += char.IsSurrogatePair(text, i) ? 2 : 1)
        {
            if (IsPythonWord(text, i) || PythonText.IsSpace(text[i]) || "_,.?!;:'…|→↓↑⁼ʰ`".IndexOf(text[i]) >= 0)
                sb.Append(text, i, char.IsSurrogatePair(text, i) ? 2 : 1);
        }
        return sb.ToString();
    }

    /// <summary><c>_chinese_to_ipa</c>: the phones of a Chinese run, read in the context of its sentence.</summary>
    public string ChineseToIpa(string text)
    {
        var data = Resources.Value;
        text = Cn2An.Transform(PythonText.Strip(text));
        foreach (var (from, to) in Punctuation) text = text.Replace(from, to);
        text = Spaces.Replace(text, "");
        text = NotKept.Replace(text, "");
        text = AroundMark.Replace(text, "$1");
        text = ToBopomofo(text);
        foreach (var (pattern, replacement) in data.LatinToBopomofo) text = pattern.Replace(text, replacement.Replace("$", "$$"));
        foreach (var (pattern, replacement) in data.BopomofoToIpa) text = pattern.Replace(text, replacement.Replace("$", "$$"));
        text = RetroflexEnd.Replace(text, "$1ɹ$2");
        text = SibilantEnd.Replace(text, "$1ɹ$2");
        text = KeepIpaCharacters(text);
        text = Bars.Replace(Marks.Replace(text, "|$1|"), "|");
        return text.TrimEnd('|');
    }
}

/// <summary>
/// MaskGCT's polyphone model (Amphion <c>BertPolyPredict</c>, <c>poly_bert_model.onnx</c>): a 6-layer, 384-wide BERT over
/// the characters after [CLS], and a linear classifier over 391 readings at each character; a polyphonic character's
/// reading is the argmax at its position.
/// </summary>
internal sealed class MaskGctPolyphoneModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly BertNetwork<T> _bert;
    private readonly DenseLayer<T> _classifier;
    private readonly List<LayerBase<T>> _layers = new();

    public MaskGctPolyphoneModel(IEngine engine, BertConfiguration? configuration = null, int classes = 391)
    {
        _engine = engine;
        Configuration = configuration ?? new BertConfiguration { VocabSize = 21128, Hidden = 384, Layers = 6, Heads = 12, Intermediate = 1536, Dropout = 0 };
        _bert = new BertNetwork<T>(engine, _layers, Configuration);
        _classifier = new DenseLayer<T>(classes, (IActivationFunction<T>)new IdentityActivation<T>());
        _layers.Add(_classifier);
        Classes = classes;
    }

    public BertConfiguration Configuration { get; }

    public int Classes { get; }

    /// <summary>The logits <c>[characters, classes]</c> of a text (the model's output after [CLS]).</summary>
    public Tensor<T> Logits(string text)
    {
        using var _ = new NoGradScope<T>();
        var ids = MaskGctMandarinFrontEnd.BertIds(text);
        var row = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) row[0, i] = ids[i];
        var hidden = _bert.Encode(row);                                                                          // [1, L, H]
        var characters = _engine.TensorSlice(_engine.Reshape(hidden, new[] { ids.Length, Configuration.Hidden }), new[] { 1, 0 },
            new[] { ids.Length - 1, Configuration.Hidden });
        return _classifier.Forward(characters);
    }

    /// <summary>The pinyin with tone digit of the character at <paramref name="index"/> ("le5").</summary>
    public string Predict(string text, int index)
    {
        var logits = Logits(text);
        int best = 0;
        for (int k = 1; k < Classes; k++)
            if (NumOps.ToDouble(logits[index, k]) > NumOps.ToDouble(logits[index, best])) best = k;
        string label = MaskGctMandarinFrontEnd.PolyphoneLabel(best + 1);
        return label.Substring(2, label.Length - 3);
    }

    /// <summary>Loads the release's <c>poly_bert_model.onnx</c>: the named BERT tensors, and the unnamed projection weights
    /// (stored transposed, in graph order: per layer the query, key, value, attention output, intermediate and output
    /// projections, then the classifier).</summary>
    public void Load(string onnxPath)
    {
        var tensors = new ONNXImporter<double>().LoadWeights(onnxPath);
        var projections = tensors.Keys.Where(k => k.StartsWith("onnx::MatMul_", StringComparison.Ordinal))
            .OrderBy(k => int.Parse(k.Substring("onnx::MatMul_".Length), System.Globalization.CultureInfo.InvariantCulture)).ToList();
        var c = Configuration;
        int expected = 6 * c.Layers + 1;
        if (projections.Count != expected) throw new InvalidDataException($"The ONNX model has {projections.Count} projection weights; expected {expected}.");
        double[] Transposed(string name, int inputs, int outputs)
        {
            var t = tensors[name];
            var result = new double[inputs * outputs];
            for (int i = 0; i < inputs; i++)
                for (int o = 0; o < outputs; o++) result[o * inputs + i] = t.GetFlat(i * outputs + o);
            return result;
        }
        var names = new Dictionary<string, double[]>(StringComparer.Ordinal);
        string[] kinds = { "attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense" };
        for (int layer = 0; layer < c.Layers; layer++)
            for (int k = 0; k < 6; k++)
            {
                int inputs = k == 5 ? c.Intermediate : c.Hidden, outputs = k == 4 ? c.Intermediate : c.Hidden;
                names[$"encoder.layer.{layer}.{kinds[k]}.weight"] = Transposed(projections[6 * layer + k], inputs, outputs);
            }
        double[] Read(string name, int[] shape)
        {
            if (names.TryGetValue(name, out var values)) return values;
            if (!tensors.TryGetValue("bert." + name, out var tensor)) throw new InvalidDataException($"The ONNX model has no tensor 'bert.{name}'.");
            return tensor.ToArray();
        }
        _bert.LoadTorchWeights(Read);
        TorchParameters.Linear(_engine, _classifier, c.Hidden, Classes, Transposed(projections[^1], c.Hidden, Classes),
            tensors["classifier.bias"].ToArray());
    }
}
