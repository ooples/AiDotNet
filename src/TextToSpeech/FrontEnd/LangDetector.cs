using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// langdetect 1.0.9 (Shuyo's language-detection, ported to Python), which MegaTTS 3 classifies its input text with: the
/// text cleaned of URLs and e-mail addresses, Vietnamese diacritics composed, spaces collapsed and (when mostly non-Latin)
/// Latin letters dropped; its character 1-, 2- and 3-grams normalized (punctuation to spaces, kana, Hangul and Bopomofo to
/// one representative each, Kanji to their class's) and skipped inside capitalized words; then seven trials that each
/// multiply the 55 languages' probabilities by randomly chosen n-grams' smoothed frequencies until one passes 0.99999,
/// averaged.
/// </summary>
/// <remarks>
/// The trials draw from Python's Mersenne Twister; langdetect seeds it from the operating system unless
/// <c>DetectorFactory.seed</c> is set, and this port always takes a seed (by default 0), so its answers repeat. The
/// profiles are langdetect's own, embedded in file-name order (the language order, which langdetect takes from the
/// directory listing).
/// </remarks>
internal sealed class LangDetector
{
    private const double AlphaDefault = 0.5, AlphaWidth = 0.05, ProbabilityThreshold = 0.1, ConvergenceThreshold = 0.99999;
    private const int IterationLimit = 1000, Trials = 7, BaseFrequency = 10000;

    private static readonly Regex Url = new("https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}", RegexOptions.CultureInvariant);
    private static readonly Regex Mail = new("[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}", RegexOptions.CultureInvariant);

    private sealed class Data
    {
        public string[] Languages = Array.Empty<string>();
        public Dictionary<string, double[]> WordProbabilities = new(StringComparer.Ordinal);
        public Dictionary<string, (int Start, int End)> Blocks = new(StringComparer.Ordinal);
        public string Latin1Excluded = "";
        public Dictionary<int, int> CjkMap = new();
        public Regex VietnameseMarks = null!;
        public string VietnameseLetters = "", DiacriticalMarks = "";
        public string[] VietnameseComposed = Array.Empty<string>();
    }

    private static readonly Lazy<Data> Shared = new(Load, isThreadSafe: true);

    private static Data Load()
    {
        using var stream = typeof(LangDetector).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.langdetect_profiles.json.gz")
            ?? throw new InvalidOperationException("The langdetect profiles are not embedded.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var root = JObject.Parse(reader.ReadToEnd());
        var data = new Data();
        var profiles = (JArray)root["profiles"]!;
        data.Languages = profiles.Select(p => (string)p["name"]!).ToArray();
        int count = profiles.Count;
        for (int index = 0; index < count; index++)
        {
            var nWords = profiles[index]["n_words"]!.Select(v => (long)v).ToArray();
            foreach (var entry in ((JObject)profiles[index]["freq"]!).Properties())
            {
                if (!data.WordProbabilities.TryGetValue(entry.Name, out var row))
                    data.WordProbabilities[entry.Name] = row = new double[count];
                int length = CodePointCount(entry.Name);
                if (length >= 1 && length <= 3) row[index] = 1.0 * (long)entry.Value / nWords[length - 1];
            }
        }
        foreach (var block in ((JObject)root["blocks"]!).Properties())
            data.Blocks[block.Name] = ((int)block.Value[0]!, (int)block.Value[1]!);
        var messages = (JObject)root["messages"]!;
        string Message(string key) => (string?)messages[key] ?? "!" + key + "!";
        data.Latin1Excluded = Message("NGram.LATIN1_EXCLUDE");
        foreach (var key in KanjiClasses)
        {
            var members = CodePoints(Message("NGram." + key));
            foreach (int member in members) data.CjkMap[member] = members[0];
        }
        data.VietnameseLetters = Message("TO_NORMALIZE_VI_CHARS");
        data.DiacriticalMarks = Message("DMARK_CLASS");
        data.VietnameseComposed = new[] { "0300", "0301", "0303", "0309", "0323" }.Select(s => Message("NORMALIZED_VI_CHARS_" + s)).ToArray();
        data.VietnameseMarks = new Regex("([" + Regex.Escape(data.VietnameseLetters) + "])([" + Regex.Escape(data.DiacriticalMarks) + "])", RegexOptions.CultureInvariant);
        return data;
    }

    // NGram.CJK_CLASS, in langdetect's order (a character in two classes takes the later's representative).
    private static readonly string[] KanjiClasses =
    {
        "KANJI_1_0", "KANJI_1_2", "KANJI_1_4", "KANJI_1_8", "KANJI_1_11", "KANJI_1_12", "KANJI_1_13", "KANJI_1_14", "KANJI_1_16", "KANJI_1_18",
        "KANJI_1_22", "KANJI_1_27", "KANJI_1_29", "KANJI_1_31", "KANJI_1_35", "KANJI_2_0", "KANJI_2_1", "KANJI_2_4", "KANJI_2_9", "KANJI_2_10",
        "KANJI_2_11", "KANJI_2_12", "KANJI_2_13", "KANJI_2_15", "KANJI_2_16", "KANJI_2_18", "KANJI_2_21", "KANJI_2_22", "KANJI_2_23", "KANJI_2_28",
        "KANJI_2_29", "KANJI_2_30", "KANJI_2_31", "KANJI_2_32", "KANJI_2_35", "KANJI_2_36", "KANJI_2_37", "KANJI_2_38", "KANJI_3_1", "KANJI_3_2",
        "KANJI_3_3", "KANJI_3_4", "KANJI_3_5", "KANJI_3_8", "KANJI_3_9", "KANJI_3_11", "KANJI_3_12", "KANJI_3_13", "KANJI_3_15", "KANJI_3_16",
        "KANJI_3_18", "KANJI_3_19", "KANJI_3_22", "KANJI_3_23", "KANJI_3_27", "KANJI_3_29", "KANJI_3_30", "KANJI_3_31", "KANJI_3_32", "KANJI_3_35",
        "KANJI_3_36", "KANJI_3_37", "KANJI_3_38", "KANJI_4_0", "KANJI_4_9", "KANJI_4_10", "KANJI_4_16", "KANJI_4_17", "KANJI_4_18", "KANJI_4_22",
        "KANJI_4_24", "KANJI_4_28", "KANJI_4_34", "KANJI_4_39", "KANJI_5_10", "KANJI_5_11", "KANJI_5_12", "KANJI_5_13", "KANJI_5_14", "KANJI_5_18",
        "KANJI_5_26", "KANJI_5_29", "KANJI_5_34", "KANJI_5_39", "KANJI_6_0", "KANJI_6_3", "KANJI_6_9", "KANJI_6_10", "KANJI_6_11", "KANJI_6_12",
        "KANJI_6_16", "KANJI_6_18", "KANJI_6_20", "KANJI_6_21", "KANJI_6_22", "KANJI_6_23", "KANJI_6_25", "KANJI_6_28", "KANJI_6_29", "KANJI_6_30",
        "KANJI_6_32", "KANJI_6_34", "KANJI_6_35", "KANJI_6_37", "KANJI_6_39", "KANJI_7_0", "KANJI_7_3", "KANJI_7_6", "KANJI_7_7", "KANJI_7_9",
        "KANJI_7_11", "KANJI_7_12", "KANJI_7_13", "KANJI_7_16", "KANJI_7_18", "KANJI_7_19", "KANJI_7_20", "KANJI_7_21", "KANJI_7_23", "KANJI_7_25",
        "KANJI_7_28", "KANJI_7_29", "KANJI_7_32", "KANJI_7_33", "KANJI_7_35", "KANJI_7_37",
    };

    /// <summary>The languages, in profile order.</summary>
    public static IReadOnlyList<string> Languages => Shared.Value.Languages;

    /// <summary>The smoothing constant (<c>set_alpha</c>; 0.5).</summary>
    public double Alpha { get; set; } = AlphaDefault;

    /// <summary>The random seed of the trials (<c>DetectorFactory.seed</c>).</summary>
    public long Seed { get; set; }

    /// <summary>The most characters of the text read (<c>set_max_text_length</c>; 10,000).</summary>
    public int MaxTextLength { get; set; } = 10000;

    /// <summary><c>detect</c>: the most probable language, or "unknown" when none passes 0.1.</summary>
    /// <exception cref="InvalidOperationException">The text has no features (langdetect's "No features in text.").</exception>
    public string Detect(string text)
    {
        var probabilities = Probabilities(text);
        return probabilities.Count > 0 ? probabilities[0].Language : "unknown";
    }

    /// <summary><c>get_probabilities</c>: the languages above 0.1, most probable first.</summary>
    public List<(string Language, double Probability)> Probabilities(string text)
    {
        var data = Shared.Value;
        var ngrams = ExtractNgrams(Clean(Append(text)));
        if (ngrams.Count == 0) throw new InvalidOperationException("No features in text.");
        int languages = data.Languages.Length;
        var total = new double[languages];
        var random = new PythonRandom(Seed);
        for (int trial = 0; trial < Trials; trial++)
        {
            var prob = Enumerable.Repeat(1.0 / languages, languages).ToArray();
            double alpha = Alpha + random.Gauss(0.0, 1.0) * AlphaWidth;
            for (int i = 0; ; i++)
            {
                var row = data.WordProbabilities[random.Choice(ngrams)];
                double weight = alpha / BaseFrequency;
                for (int l = 0; l < languages; l++) prob[l] *= weight + row[l];
                if (i % 5 == 0 && (Normalize(prob) > ConvergenceThreshold || i >= IterationLimit)) break;
            }
            for (int l = 0; l < languages; l++) total[l] += prob[l] / Trials;
        }
        // list.sort(reverse=True) on the probabilities: descending, ties in language order.
        return data.Languages.Select((language, l) => (Language: language, Probability: total[l]))
            .Where(p => p.Probability > ProbabilityThreshold).OrderByDescending(p => p.Probability).ToList();
    }

    private static double Normalize(double[] prob)
    {
        double max = 0.0, sum = 0.0;
        foreach (var p in prob) sum += p;
        for (int i = 0; i < prob.Length; i++)
        {
            double p = prob[i] / sum;
            if (max < p) max = p;
            prob[i] = p;
        }
        return max;
    }

    // Detector.append: URLs and e-mail addresses to spaces, Vietnamese composed, runs of spaces to one, cut to the
    // length limit (in code points).
    private List<int> Append(string text)
    {
        text = Url.Replace(text, " ");
        text = Mail.Replace(text, " ");
        text = NormalizeVietnamese(text);
        var points = CodePoints(text);
        var result = new List<int>();
        int previous = 0;
        for (int i = 0; i < Math.Min(points.Count, MaxTextLength); i++)
        {
            int ch = points[i];
            if (ch != ' ' || previous != ' ') result.Add(ch);
            previous = ch;
        }
        return result;
    }

    /// <summary><c>NGram.normalize_vi</c>: a vowel and a combining diacritic to the precomposed letter.</summary>
    internal static string NormalizeVietnamese(string text)
    {
        var data = Shared.Value;
        return data.VietnameseMarks.Replace(text, m =>
            data.VietnameseComposed[data.DiacriticalMarks.IndexOf(m.Groups[2].Value, StringComparison.Ordinal)][data.VietnameseLetters.IndexOf(m.Groups[1].Value, StringComparison.Ordinal)].ToString());
    }

    // Detector.cleaning_text: with more than twice as many characters from U+0300 on as Latin ones ('A'..'z'), the
    // Latin ones dropped. (langdetect compares the block number with a block name there, which never matches, so every
    // character from U+0300 on counts.)
    private static List<int> Clean(List<int> text)
    {
        int latin = 0, other = 0;
        foreach (int ch in text)
        {
            if (ch >= 'A' && ch <= 'z') latin++;
            else if (ch >= 0x300) other++;
        }
        return latin * 2 < other ? text.Where(ch => ch < 'A' || ch > 'z').ToList() : text;
    }

    /// <summary><c>Detector._extract_ngrams</c> after cleaning: every known 1-, 2- and 3-gram, outside capitalized words.</summary>
    internal List<string> ExtractNgrams(string text) => ExtractNgrams(Clean(Append(text)));

    private static List<string> ExtractNgrams(List<int> text)
    {
        var map = Shared.Value.WordProbabilities;
        var result = new List<string>();
        var grams = new List<int> { ' ' };
        bool capitalWord = false;
        foreach (int raw in text)
        {
            // NGram.add_char
            int ch = NormalizeChar(raw);
            int last = grams[^1];
            if (last == ' ')
            {
                grams.Clear();
                grams.Add(' ');
                capitalWord = false;
                if (ch == ' ') continue;
            }
            else if (grams.Count >= 3) grams.RemoveAt(0);
            grams.Add(ch);
            capitalWord = IsUpper(ch) ? capitalWord || IsUpper(last) : false;
            if (capitalWord) continue;
            for (int n = 1; n <= 3; n++)
            {
                if (grams.Count < n) break;
                var w = FromCodePoints(grams.Skip(grams.Count - n));
                if (w != " " && map.ContainsKey(w)) result.Add(w);
            }
        }
        return result;
    }

    /// <summary><c>NGram.normalize</c> of one character.</summary>
    internal static int NormalizeChar(int ch)
    {
        var data = Shared.Value;
        bool In(string block) => data.Blocks[block] is var (start, end) && ch >= start && ch <= end;
        if (In("BASIC_LATIN")) return ch < 'A' || (ch > 'Z' && ch < 'a') || ch > 'z' ? ' ' : ch;
        if (In("LATIN_1_SUPPLEMENT")) return data.Latin1Excluded.IndexOf((char)ch) >= 0 ? ' ' : ch;
        if (In("LATIN_EXTENDED_B")) return ch == 0x219 ? 0x15f : ch == 0x21b ? 0x163 : ch;
        if (In("GENERAL_PUNCTUATION")) return ' ';
        if (In("ARABIC")) return ch == 0x6cc ? 0x64a : ch;
        if (In("LATIN_EXTENDED_ADDITIONAL")) return ch >= 0x1ea0 ? 0x1ec3 : ch;
        if (In("HIRAGANA")) return 0x3042;
        if (In("KATAKANA")) return 0x30a2;
        if (In("BOPOMOFO") || In("BOPOMOFO_EXTENDED")) return 0x3105;
        if (In("CJK_UNIFIED_IDEOGRAPHS")) return data.CjkMap.TryGetValue(ch, out var representative) ? representative : ch;
        if (In("HANGUL_SYLLABLES")) return 0xac00;
        return ch;
    }

    // str.isupper() of one character: an uppercase letter or Other_Uppercase (Roman numerals, circled and squared letters).
    private static bool IsUpper(int ch) =>
        (ch >= 0x2160 && ch <= 0x216f) || (ch >= 0x24b6 && ch <= 0x24cf) || (ch >= 0x1f130 && ch <= 0x1f149) || (ch >= 0x1f150 && ch <= 0x1f169)
        || (ch >= 0x1f170 && ch <= 0x1f189)
        || (!(ch >= 0xd800 && ch <= 0xdfff) && CharUnicodeInfo.GetUnicodeCategory(char.ConvertFromUtf32(ch), 0) == UnicodeCategory.UppercaseLetter);

    private static List<int> CodePoints(string text)
    {
        var points = new List<int>(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])) { points.Add(char.ConvertToUtf32(text[i], text[i + 1])); i++; }
            else points.Add(text[i]);
        }
        return points;
    }

    private static int CodePointCount(string text) => CodePoints(text).Count;

    private static string FromCodePoints(IEnumerable<int> points)
    {
        var builder = new StringBuilder();
        foreach (int p in points)
        {
            if (p >= 0x10000) builder.Append(char.ConvertFromUtf32(p));
            else builder.Append((char)p);
        }
        return builder.ToString();
    }
}
