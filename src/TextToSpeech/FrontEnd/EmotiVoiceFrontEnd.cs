using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// EmotiVoice's bilingual text front end (netease-youdao/EmotiVoice <c>frontend.py</c>, <c>frontend_cn.py</c>,
/// <c>frontend_en.py</c>): Chinese and English text to the released model's phoneme tokens.
/// </summary>
/// <remarks>
/// <para>
/// As the release runs it (<c>g2p_cn_en</c>): Arabic numerals become Chinese numerals (EmotiVoice's copy of cn2an), the
/// text splits into Chinese runs (Han characters, Chinese punctuation and spaces) and other runs. Chinese runs are cut by
/// jieba; pypinyin reads each word in TONE3 style with pypinyin-dict's CC-CEDICT phrases loaded, and each syllable splits
/// into an initial and a final (<c>split_py</c>), syllables joined by <c>sp0</c>, words by <c>sp1</c>, a punctuation mark
/// replacing the boundary before it with <c>sp3</c>. Other runs are English: words from LibriSpeech's lexicon, or, for
/// words it lacks, from g2p_en, each phone in brackets, words joined by <c>engsp1</c> and <c>engsp4</c> at punctuation;
/// <c>cn_eng_sp</c> and <c>eng_cn_sp</c> mark a change of language. The sequence is framed by <c>&lt;sos/eos&gt;</c>.
/// </para>
/// <para>The release's quirks are kept, for the model was trained on them: a punctuation mark at a Chinese run's start
/// removes the frame before it, an English word g2p_en reads with a leading mark (a quote) is dropped, and
/// <c>yue</c> splits as <c>y ve</c>.</para>
/// <para><b>For Beginners:</b> This turns written Chinese and English into the sound symbols EmotiVoice was trained on,
/// such as <c>n i3 sp0 h ao3</c> for 你好.</para>
/// </remarks>
public sealed class EmotiVoiceFrontEnd
{
    private static readonly Lazy<EmotiVoiceFrontEnd> SharedInstance = new(() => new EmotiVoiceFrontEnd(), isThreadSafe: true);

    private readonly Dictionary<string, string[]> _lexicon = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _tokens = new(StringComparer.Ordinal);
    private readonly List<string> _tokenList = new();
    private readonly List<string> _speakers = new();

    private EmotiVoiceFrontEnd()
    {
        foreach (var line in ChineseText.ReadLines("emotivoice_lexicon.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            _lexicon[line.Substring(0, tab)] = line.Substring(tab + 1).Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        }
        foreach (var token in ChineseText.ReadLines("emotivoice_tokens.txt.gz"))
        {
            _tokens[token] = _tokenList.Count;
            _tokenList.Add(token);
        }
        _speakers.AddRange(ChineseText.ReadLines("emotivoice_speakers.txt.gz"));
    }

    /// <summary>The shared instance (the dictionaries load once, on first use).</summary>
    public static EmotiVoiceFrontEnd Default => SharedInstance.Value;

    /// <summary>The released token list (502 tokens; a phoneme's id is its index).</summary>
    public IReadOnlyList<string> Tokens => _tokenList;

    /// <summary>The released speaker list (a speaker's id is its index).</summary>
    public IReadOnlyList<string> Speakers => _speakers;

    /// <summary>The id of a speaker of the released list, by its name (such as "8051").</summary>
    public int SpeakerId(string name)
    {
        int id = _speakers.IndexOf(name ?? throw new ArgumentNullException(nameof(name)));
        if (id < 0) throw new ArgumentException($"'{name}' is not a speaker of the released list.", nameof(name));
        return id;
    }

    /// <summary>The token ids of <paramref name="text"/>.</summary>
    public int[] TokenIds(string text) =>
        Phonemes(text).Select(p => _tokens.TryGetValue(p, out int id) ? id
            : throw new ArgumentException($"The phoneme '{p}' of \"{text}\" is not in EmotiVoice's token list.", nameof(text))).ToArray();

    // ---------------------------------------------------------------- frontend.py

    private static readonly Regex Parts = new(
        "([^一-龥]+|[ 。，？！；：“”‘’《》〈〉【】『』—…、（）一-龥]+)",
        RegexOptions.IgnoreCase);
    private static readonly Regex Chinese = new("[一-龥]");
    private static readonly Regex Digits = new(@"(\d[\d\.]*)");

    /// <summary><c>g2p_cn_en</c>: the phonemes of <paramref name="text"/>, framed by <c>&lt;sos/eos&gt;</c>.</summary>
    public IReadOnlyList<string> Phonemes(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        text = NumbersToChinese(text);
        var tts = new List<string> { "<sos/eos>" };
        string chartype = "";
        bool textHasChinese = Chinese.IsMatch(text);
        foreach (Match part in Parts.Matches(text))
        {
            string p = part.Value;
            if (p == " ") continue;
            IEnumerable<string> phoneme;
            if (StartsWithDigit(p) && (textHasChinese || chartype.Length == 0) || Chinese.IsMatch(p))
            {
                if (chartype == "en") tts.Add("eng_cn_sp");
                var cn = Split(ChinesePhonemes(p));
                phoneme = cn.Skip(1).Take(cn.Count - 2).ToList();
                chartype = "cn";
            }
            else if (Parts.Match(p) is { Success: true, Index: 0 })
            {
                if (chartype == "cn" && !tts[tts.Count - 1].Contains("sp")) tts.Add("cn_eng_sp");
                var en = Split(EnglishPhonemes(p));
                if (en.Count == 0) continue;
                phoneme = en;
                chartype = "en";
            }
            else
            {
                continue;
            }
            tts.AddRange(phoneme);
        }
        tts = Split(string.Join(" ", tts));
        if (tts[tts.Count - 1].Contains("sp")) tts.RemoveAt(tts.Count - 1);
        tts.Add("<sos/eos>");
        return tts;
    }

    // re_digits.match: a digit at the start (Python's \d, any Unicode digit).
    private static bool StartsWithDigit(string text) => text.Length > 0 && char.IsDigit(text[0]);

    private static List<string> Split(string text) => text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).ToList();

    // ---------------------------------------------------------------- frontend_cn.py

    /// <summary><c>tn_chinese</c>: each run of digits and periods starting with a digit, as Chinese numerals.</summary>
    public static string NumbersToChinese(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Digits.Replace(text, m => MandarinG2P.AnToCn(m.Value, joinThousands: false));
    }

    private const string ChinesePunctuation = "。，？！；：“”‘’《》〈〉【】『』—…、（）";
    private const string AsciiPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    // g2p_cn. has_english_punctuation is `text in string.punctuation`, a substring test ("()" counts).
    private static string ChinesePhonemes(string text)
    {
        var result = new List<string> { "<sos/eos>" };
        foreach (var segment in Jieba.Default.Cut(text))
        {
            if (segment == " ") continue;
            var readings = PypinyinReader.EmotiVoiceTone3.Read(NumbersToChinese(segment)).ToList();
            if (readings.Any(r => r.IndexOfAny(ChinesePunctuation.ToCharArray()) >= 0) || readings.Any(r => AsciiPunctuation.IndexOf(r, StringComparison.Ordinal) >= 0))
            {
                result.RemoveAt(result.Count - 1);
                result.Add("sp3");
            }
            else
            {
                result.Add(string.Join(" sp0 ", readings.Select(r =>
                {
                    var (initial, final) = SplitPinyin(r, text);
                    return initial + " " + final;
                })));
                result.Add("sp1");
            }
        }
        result.Add("<sos/eos>");
        return string.Join(" ", result);
    }

    private static readonly Regex SpecialPinyin = new("^(n|ng|m)$");
    private static readonly Regex Vowel = new("a|e|i|o|u|v");

    /// <summary><c>split_py</c>: a TONE3 syllable as its initial and its final with the tone.</summary>
    internal static (string Initial, string Final) SplitPinyin(string py, string text)
    {
        if (py.Length < 2)
            throw new ArgumentException($"\"{text}\" holds '{py}', which EmotiVoice's front end cannot read (it is neither a Chinese character nor punctuation).", nameof(text));
        string tone = py.Substring(py.Length - 1);
        py = py.Substring(0, py.Length - 1);
        string sm, ym, suffix = "";
        if (SpecialPinyin.IsMatch(py)) py = "e" + py;
        if (py.EndsWith("r", StringComparison.Ordinal))
        {
            suffix = "r";
            py = py.Substring(0, py.Length - 1);
        }
        if (py is "zi" or "ci" or "si" or "ri")
        {
            sm = py.Substring(0, 1);
            ym = "ii";
        }
        else if (py is "zhi" or "chi" or "shi")
        {
            sm = py.Substring(0, 2);
            ym = "iii";
        }
        else if (py is "ya" or "yan" or "yang" or "yao" or "ye" or "yong" or "you")
        {
            sm = "";
            ym = "i" + py.Substring(1);
        }
        else if (py is "yi" or "yin" or "ying")
        {
            sm = "";
            ym = py.Substring(1);
        }
        // The release compares with 'yue ' (a trailing space), so yue falls through to the general case: y ve.
        else if (py is "yu" or "yv" or "yuan" or "yvan" or "yve" or "yun" or "yvn")
        {
            sm = "";
            ym = "v" + py.Substring(2);
        }
        else if (py == "wu")
        {
            sm = "";
            ym = "u";
        }
        else if (py.Length > 0 && py[0] == 'w')
        {
            sm = "";
            ym = "u" + py.Substring(1);
        }
        else if (py.Length >= 2 && (py[0] == 'j' || py[0] == 'q' || py[0] == 'x') && py[1] == 'u')
        {
            sm = py.Substring(0, 1);
            ym = "v" + py.Substring(2);
        }
        else
        {
            var vowel = Vowel.Match(py);
            if (!vowel.Success)
                throw new ArgumentException($"\"{text}\" holds '{py}{tone}', which EmotiVoice's front end cannot read.", nameof(text));
            sm = py.Substring(0, vowel.Index);
            ym = py.Substring(vowel.Index);
            ym = ym switch { "ui" => "uei", "iu" => "iou", "un" => "uen", "ue" => "ve", _ => ym };
        }
        return (sm, ym + suffix + tone);
    }

    // ---------------------------------------------------------------- frontend_en.py

    private static readonly Regex EnglishSplit = new(@"([,;.\-\?\!\s+])");

    // get_eng_phoneme(text, g2p, lexicon, pad_sos_eos=False)
    private string EnglishPhonemes(string text)
    {
        var phones = new List<string>();
        foreach (var w in EnglishSplit.Split(text).Where(x => x.Length > 0 && x != " "))
        {
            if (_lexicon.TryGetValue(PythonLower(w), out var entry))
            {
                foreach (var ph in entry)
                    if (ph is not ("," or " " or "'")) phones.Add("[" + ph + "]");
                if (phones.Count > 0 && !phones[phones.Count - 1].Contains("sp")) phones.Add("engsp1");
            }
            else
            {
                var phone = G2pEn.Default.Word(w);
                if (phone.Count == 0) continue;
                if (phone[0].Length > 0 && phone[0].All(char.IsLetterOrDigit))
                {
                    foreach (var ph in phone)
                    {
                        if (ph is not ("," or " " or "'")) phones.Add("[" + ph + "]");
                        if (ph == " " && !phones[phones.Count - 1].Contains("sp")) phones.Add("engsp1");
                    }
                }
                else if (phones.Count > 0)
                {
                    phones.RemoveAt(phones.Count - 1);
                    phones.Add("engsp4");
                }
            }
        }
        if (phones.Count > 0 && phones[phones.Count - 1].Contains("engsp")) phones.RemoveAt(phones.Count - 1);
        return string.Join(" ", phones);
    }

    // str.lower() for the lexicon's keys (lower-cased by read_lexicon the same way).
    private static string PythonLower(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (char c in text) builder.Append(char.ToLowerInvariant(c));
        return builder.ToString();
    }
}
