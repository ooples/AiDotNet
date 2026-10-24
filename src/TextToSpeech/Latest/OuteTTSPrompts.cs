using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>A word of an OuteTTS speaker profile: its text, its duration, its two DAC codebooks' codes and its features
/// (each 0–100).</summary>
public sealed class OuteTTSWord
{
    /// <summary>The word as written.</summary>
    [JsonProperty("word")] public string Word { get; set; } = "";

    /// <summary>Its duration in seconds (its frames at 75 per second, rounded to 0.01).</summary>
    [JsonProperty("duration")] public double Duration { get; set; }

    /// <summary>The first codebook's codes.</summary>
    [JsonProperty("c1")] public int[] C1 { get; set; } = Array.Empty<int>();

    /// <summary>The second codebook's codes.</summary>
    [JsonProperty("c2")] public int[] C2 { get; set; } = Array.Empty<int>();

    /// <summary>Its energy, spectral centroid and pitch, each scaled to 0–100.</summary>
    [JsonProperty("features")] public OuteTTSFeatures Features { get; set; } = new();
}

/// <summary>OuteTTS's audio features of a stretch of speech, each scaled to 0–100.</summary>
public sealed class OuteTTSFeatures
{
    /// <summary>RMS energy × 100.</summary>
    [JsonProperty("energy")] public int Energy { get; set; }

    /// <summary>The spectral centroid as a fraction of the Nyquist frequency, × 100.</summary>
    [JsonProperty("spectral_centroid")] public int SpectralCentroid { get; set; }

    /// <summary>The mean pitch between 75 and 600 Hz as a fraction of that range, × 100.</summary>
    [JsonProperty("pitch")] public int Pitch { get; set; }
}

/// <summary>An OuteTTS 1.0 speaker profile (the release's speaker JSON): the transcript of a short recording, its words
/// with their codes and features, and the whole recording's features.</summary>
public sealed class OuteTTSSpeaker
{
    /// <summary>The transcript.</summary>
    [JsonProperty("text")] public string Text { get; set; } = "";

    /// <summary>The words, in order.</summary>
    [JsonProperty("words")] public List<OuteTTSWord> Words { get; set; } = new();

    /// <summary>The recording's features.</summary>
    [JsonProperty("global_features")] public OuteTTSFeatures GlobalFeatures { get; set; } = new();

    /// <summary>The release's interface version (3 for OuteTTS 1.0).</summary>
    [JsonProperty("interface_version")] public int InterfaceVersion { get; set; } = 3;

    /// <summary>Reads a speaker saved by the release (<c>interface.save_speaker</c>) or by <see cref="ToJson"/>.</summary>
    public static OuteTTSSpeaker FromJson(string json)
    {
        var speaker = JsonConvert.DeserializeObject<OuteTTSSpeaker>(json ?? throw new ArgumentNullException(nameof(json)))
            ?? throw new InvalidDataException("The speaker JSON is empty.");
        if (speaker.InterfaceVersion != 3)
            throw new InvalidDataException($"Speaker interface version mismatch: {speaker.InterfaceVersion} != 3.");
        return speaker;
    }

    /// <summary>The speaker as the release's JSON.</summary>
    public string ToJson() => JsonConvert.SerializeObject(this, Formatting.Indented);

    internal OuteTTSSpeaker Clone() => FromJson(ToJson());

    /// <summary>The release's bundled speaker (<c>en-female-1-neutral</c>).</summary>
    public static OuteTTSSpeaker DefaultEnglishFemale()
    {
        using var stream = typeof(OuteTTSSpeaker).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.Latest.outetts_en_female_1_neutral.json")
            ?? throw new InvalidDataException("The embedded resource outetts_en_female_1_neutral.json is missing.");
        using var reader = new StreamReader(stream);
        return FromJson(reader.ReadToEnd());
    }
}

/// <summary>
/// OuteTTS 1.0's prompt format and text handling (edwko/OuteTTS <c>version/v3/prompt_processor.py</c> and
/// <c>utils/chunking.py</c>): text normalization (ftfy, NFKC, quotes, dashes, spacing), merging a speaker's transcript
/// with the text, the completion and training prompts with each word's features, duration and codes, and the 10–30 word
/// chunks long text is generated in (MeCab for Japanese and Chinese).
/// </summary>
internal static class OuteTTSPrompts
{
    public const string Bos = "<|im_start|>";
    public const string Eos = "<|im_end|>";
    public const string TextStart = "<|text_start|>";
    public const string TextEnd = "<|text_end|>";
    public const string AudioStart = "<|audio_start|>";
    public const string AudioEnd = "<|audio_end|>";
    public const string WordStart = "<|word_start|>";
    public const string WordEnd = "<|word_end|>";
    public const string FeaturesToken = "<|features|>";
    public const string Code = "<|code|>";
    public const string GlobalFeaturesStart = "<|global_features_start|>";
    public const string GlobalFeaturesEnd = "<|global_features_end|>";

    // Python's \w and \s (str.isalnum() or '_', str.isspace()), exported with ftfy's tables.
    private static readonly Lazy<(string Word, string Space)> Classes = new(() => Ftfy.PythonClasses, isThreadSafe: true);

    private static Regex? _spaces, _spaceBeforePunctuation, _spaceAfterPunctuation, _apostropheWords, _apostropheEnd, _contraction;

    private static void BuildRegexes()
    {
        if (_contraction is not null) return;
        var (w, s) = Classes.Value;
        string sInner = s.Substring(1, s.Length - 2);
        var options = RegexOptions.CultureInvariant;
        _spaces = new Regex(s + "+", options);
        _spaceBeforePunctuation = new Regex(s + "+([,.?!:;])", options);
        _spaceAfterPunctuation = new Regex("([,.?!:;])(?=[^" + sInner + ",.?!:;])", options);
        _apostropheWords = new Regex("(" + w + ")" + s + "+'" + s + "*(" + w + ")", options);
        _apostropheEnd = new Regex("(" + w + ")" + s + "+'" + s + "*([,.?!:;" + sInner + "]|$)", options);
        _contraction = new Regex("(" + w + ")" + s + "*'" + s + "*([tsdmreSNTDMRE])(?!" + w + ")", options | RegexOptions.IgnoreCase);
    }

    private static readonly Regex Ellipsis = new(@"\.{2,}", RegexOptions.CultureInvariant);
    private static readonly Regex DoubleQuotes = new("[“”„‟«»]", RegexOptions.CultureInvariant);
    private static readonly Regex SingleQuotes = new("[‘’‛‹›`´]", RegexOptions.CultureInvariant);
    private static readonly Regex Dashes = new("[–—―−‐]", RegexOptions.CultureInvariant);
    private static readonly Regex DashRuns = new("-{2,}", RegexOptions.CultureInvariant);
    private static readonly Regex Controls = new("[\u0000-\u001f\u007f-\u009f\u00ad\u200b-\u200d\ufeff]", RegexOptions.CultureInvariant);
    private static readonly Regex RepeatedMarks = new(@"([?!])\1+", RegexOptions.CultureInvariant);

    /// <summary><c>PromptProcessor.text_normalizations</c>.</summary>
    public static string NormalizeText(string text)
    {
        if (text is null) return "";
        BuildRegexes();
        text = Ftfy.FixText(text);
        text = text.Normalize(System.Text.NormalizationForm.FormKC);
        text = text.Replace("…", "...");
        text = Ellipsis.Replace(text, "...");
        text = DoubleQuotes.Replace(text, "\"");
        text = SingleQuotes.Replace(text, "'");
        text = Dashes.Replace(text, "-");
        text = DashRuns.Replace(text, "-");
        text = Controls.Replace(text, "");
        text = _spaces!.Replace(text, " ");
        text = _spaceBeforePunctuation!.Replace(text, "$1");
        text = _spaceAfterPunctuation!.Replace(text, "$1 ");
        text = _apostropheWords!.Replace(text, "$1'$2");
        text = _apostropheEnd!.Replace(text, "$1'$2");
        text = _contraction!.Replace(text, "$1'$2");
        text = RepeatedMarks.Replace(text, "$1");
        text = _spaces.Replace(text, " ");
        return PythonStrip(text.Replace("\"", ""));
    }

    // str.strip(): the characters str.isspace() accepts, from both ends.
    internal static string PythonStrip(string text)
    {
        BuildRegexes();
        var (_, s) = Classes.Value;
        return Regex.Replace(Regex.Replace(text, "^" + s + "+", ""), s + "+$", "");
    }

    // _get_separator
    private static string Separator(string text)
    {
        bool kana = text.Any(c => c >= '぀' && c <= 'ヿ');
        bool han = text.Any(c => c >= '一' && c <= '鿿');
        return kana || han ? "。" : ". ";
    }

    /// <summary><c>merge_speaker_text</c>: the speaker's transcript, a separator when it does not end a sentence, then the
    /// text; and the separator (stripped) the speaker's last word takes.</summary>
    public static (string Text, string Separator) MergeSpeakerText(string input, string speakerText)
    {
        speakerText = PythonStrip(speakerText);
        string separator = Separator(speakerText);
        var allowed = separator == "。" ? new[] { "。", "？", "！", "?", "!" } : new[] { ".", "?", "!" };
        string rs = "";
        if (speakerText.Length > 0)
        {
            string last = speakerText.Substring(speakerText.Length - 1);
            if (!allowed.Contains(last)) rs = separator;
            else if (separator != "。") rs = " ";
        }
        return (PythonStrip(speakerText) + rs + PythonStrip(input), PythonStrip(rs));
    }

    private static string Features(OuteTTSFeatures f) =>
        $"<|energy_{f.Energy.ToString(CultureInfo.InvariantCulture)}|><|spectral_centroid_{f.SpectralCentroid.ToString(CultureInfo.InvariantCulture)}|>" +
        $"<|pitch_{f.Pitch.ToString(CultureInfo.InvariantCulture)}|>";

    /// <summary><c>create_codes</c>: each word with its features, duration and code pairs, one per line.</summary>
    public static string CreateCodes(IEnumerable<OuteTTSWord> words)
    {
        var lines = new List<string>();
        foreach (var w in words)
        {
            var line = new System.Text.StringBuilder();
            line.Append(WordStart).Append(w.Word).Append(FeaturesToken).Append("<|t_").Append(FormatFixed(w.Duration, 2)).Append("|>");
            line.Append(Features(w.Features)).Append(Code);
            for (int i = 0; i < w.C1.Length; i++)
                line.Append("<|c1_").Append(w.C1[i].ToString(CultureInfo.InvariantCulture)).Append("|><|c2_")
                    .Append(w.C2[i].ToString(CultureInfo.InvariantCulture)).Append("|>");
            line.Append(WordEnd);
            lines.Add(line.ToString());
        }
        return string.Join("\n", lines);
    }

    private static string Header(string text) => $"{Bos}\n{TextStart}{text}{TextEnd}\n{AudioStart}\n";

    /// <summary><c>get_completion_prompt</c>: the prompt generation continues, with the speaker's words as the start of the
    /// audio when there is a speaker.</summary>
    public static string CompletionPrompt(string text, OuteTTSSpeaker? speaker)
    {
        text = NormalizeText(text);
        if (speaker is null) return Header(text);
        speaker = speaker.Clone();
        var (merged, separator) = MergeSpeakerText(text, speaker.Text);
        if (speaker.Words.Count == 0) throw new ArgumentException("The speaker has no words.", nameof(speaker));
        speaker.Words[speaker.Words.Count - 1].Word += separator;
        return Header(merged) + CreateCodes(speaker.Words) + "\n" + WordStart;
    }

    /// <summary><c>get_training_prompt</c>: a whole training example.</summary>
    public static string TrainingPrompt(OuteTTSSpeaker speaker)
    {
        string text = NormalizeText(speaker.Text);
        return Header(text) + GlobalFeaturesStart + Features(speaker.GlobalFeatures) + GlobalFeaturesEnd + "\n" + CreateCodes(speaker.Words) +
            "\n" + AudioEnd + "\n" + Eos + "\n";
    }

    /// <summary>Python's <c>"{:.Nf}".format(value)</c>: the exact binary value rounded half to even.</summary>
    internal static string FormatFixed(double value, int digits)
    {
        if (double.IsNaN(value) || double.IsInfinity(value)) return value.ToString(CultureInfo.InvariantCulture);
        bool negative = value < 0 || (value == 0 && double.IsNegative(value));
        long bits = BitConverter.DoubleToInt64Bits(Math.Abs(value));
        int exponent = (int)((bits >> 52) & 0x7ff);
        long mantissa = bits & 0xfffffffffffffL;
        if (exponent == 0) exponent++; else mantissa |= 1L << 52;
        exponent -= 1075;
        // value = mantissa · 2^exponent; scaled = value · 10^digits, rounded half to even.
        BigInteger numerator = new BigInteger(mantissa) * BigInteger.Pow(10, digits);
        BigInteger denominator = BigInteger.One;
        if (exponent >= 0) numerator <<= exponent; else denominator <<= -exponent;
        BigInteger quotient = BigInteger.DivRem(numerator, denominator, out BigInteger remainder);
        int compare = (remainder * 2).CompareTo(denominator);
        if (compare > 0 || (compare == 0 && !quotient.IsEven)) quotient += 1;
        string s = quotient.ToString(CultureInfo.InvariantCulture).PadLeft(digits + 1, '0');
        string result = digits > 0 ? s.Substring(0, s.Length - digits) + "." + s.Substring(s.Length - digits) : s;
        return (negative ? "-" : "") + result;
    }

    // ---------------------------------------------------------------- chunking (utils/chunking.py)

    // check_language: any hiragana, katakana or CJK ideograph. Called on a list of tokens, the release compares each
    // token as a string with those bounds; a string compares as Python's does, by code point.
    private static bool InRange(string s, char low, char high) =>
        string.CompareOrdinal(s, low.ToString()) >= 0 && string.CompareOrdinal(s, high.ToString()) <= 0;

    internal static bool CheckLanguage(string text) => text.Any(c => (c >= '぀' && c <= 'ゟ') || (c >= '゠' && c <= 'ヿ') || (c >= '一' && c <= '鿿'));

    internal static bool CheckLanguage(IEnumerable<string> tokens) =>
        tokens.Any(t => InRange(t, '぀', 'ゟ') || InRange(t, '゠', 'ヿ') || InRange(t, '一', '鿿'));

    // str.split(): runs of whitespace (str.isspace) separate the words.
    private static List<string> PythonSplit(string text)
    {
        BuildRegexes();
        return _spaces!.Split(text).Where(t => t.Length > 0).ToList();
    }

    // tokenize_text
    internal static List<string> TokenizeText(string text)
    {
        if (PythonStrip(text).Length == 0) return new List<string>();
        return CheckLanguage(text) ? PythonSplit(MeCab.UnidicLite.Wakati(text)) : PythonSplit(text);
    }

    private static readonly Lazy<Regex> SentenceEnds = new(() => new Regex("([.!?。！？︕︖]+" + Classes.Value.Space + "*)", RegexOptions.CultureInvariant));

    // split_into_sentences
    internal static List<string> SplitIntoSentences(string text)
    {
        var parts = SentenceEnds.Value.Split(text);
        var sentences = new List<string>();
        for (int i = 0; i < parts.Length; i += 2)
        {
            string current = parts[i];
            if (i + 1 < parts.Length) current += parts[i + 1];
            if (PythonStrip(current).Length > 0) sentences.Add(PythonStrip(current));
        }
        return sentences;
    }

    /// <summary><c>chunk_text(text, min_words=10, max_words=30)</c>.</summary>
    public static List<string> ChunkText(string text, int minWords = 10, int maxWords = 30)
    {
        BuildRegexes();
        text = text.Normalize(System.Text.NormalizationForm.FormKC);
        text = PythonStrip(_spaces!.Replace(text, " "));
        var chunks = new List<string>();
        if (text.Length == 0) return chunks;
        string current = "";
        int currentCount = 0;
        foreach (var raw in SplitIntoSentences(text))
        {
            string sentence = PythonStrip(raw);
            if (sentence.Length == 0) continue;
            var tokens = TokenizeText(sentence);
            int count = tokens.Count;
            if (count > maxWords)
            {
                if (current.Length > 0)
                {
                    chunks.Add(current);
                    current = "";
                    currentCount = 0;
                }
                var part = new List<string>();
                foreach (var token in tokens)
                {
                    part.Add(token);
                    if (part.Count >= maxWords)
                    {
                        chunks.Add(CheckLanguage(sentence) ? string.Concat(part) : string.Join(" ", part));
                        part.Clear();
                    }
                }
                if (part.Count > 0) chunks.Add(CheckLanguage(sentence) ? string.Concat(part) : string.Join(" ", part));
                continue;
            }
            if (currentCount + count <= maxWords)
            {
                current = current.Length > 0 ? current + " " + sentence : sentence;
                currentCount += count;
            }
            else if (currentCount >= minWords)
            {
                chunks.Add(current);
                current = sentence;
                currentCount = count;
            }
            else
            {
                int spaceLeft = maxWords - currentCount;
                var currentPart = tokens.Take(spaceLeft).ToList();
                var remaining = tokens.Skip(spaceLeft).ToList();
                if (current.Length > 0)
                    chunks.Add(CheckLanguage(currentPart) ? current + string.Concat(currentPart) : current + " " + string.Join(" ", currentPart));
                else
                    chunks.Add(CheckLanguage(currentPart) ? string.Concat(currentPart) : string.Join(" ", currentPart));
                current = CheckLanguage(remaining) ? string.Concat(remaining) : string.Join(" ", remaining);
                currentCount = remaining.Count;
            }
        }
        if (current.Length > 0) chunks.Add(current);
        return chunks;
    }
}
