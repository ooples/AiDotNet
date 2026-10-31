using System.Globalization;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// GPT-SoVITS's Japanese front end (RVC-Boss/GPT-SoVITS at b4570b8, MIT; <c>text/japanese.py</c>, after CjangCjengh's
/// vits and ESPnet's <c>pyopenjtalk_g2p_prosody</c>): repeated punctuation collapsed; then "\uff05" read out, the text lower
/// cased and cut at every character outside Japanese, Latin letters and digits, each run of such characters read by
/// pyopenjtalk 0.4.1 (with GPT-SoVITS's user dictionary) into phonemes with the prosody marks <c>[</c> (pitch rise),
/// <c>]</c> (pitch fall), <c>#</c> (accent phrase border) and <c>_</c> (pause), and the cut characters kept between
/// them (full-width punctuation mapped to ASCII).
/// </summary>
internal sealed class GptSoVitsJapaneseFrontEnd
{
    private readonly OpenJTalk _openJTalk;

    /// <summary>Reads Japanese with the Open JTalk dictionary in <paramref name="dictionaryDirectory"/>
    /// (open_jtalk_dic_utf_8-1.11) and GPT-SoVITS's user dictionary (<c>text/ja_userdic/userdict.csv</c>).</summary>
    public GptSoVitsJapaneseFrontEnd(string dictionaryDirectory)
    {
        _openJTalk = new OpenJTalk(dictionaryDirectory, ChineseText.ReadLines("gptsovits_ja_userdict.csv.gz"));
    }

    private static readonly Regex RepeatedPunctuation = new("([!?\u2026,.\\-])([!?\u2026,.\\-])+");

    /// <summary><c>japanese.text_normalize</c>.</summary>
    public static string TextNormalize(string text) => RepeatedPunctuation.Replace(text, "$1");

    private const string JapaneseClass = "A-Za-z\\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d";
    private static readonly Regex JapaneseCharacter = new("^[" + JapaneseClass + "]");
    private static readonly Regex JapaneseMark = new("[^" + JapaneseClass + "]");

    /// <summary><c>japanese.g2p(norm_text)</c> (with prosody).</summary>
    public List<string> G2P(string text)
    {
        text = text.Replace("\uff05", "\u30d1\u30fc\u30bb\u30f3\u30c8");
        text = PythonText.Lower(text);
        var sentences = JapaneseMark.Split(text);
        var marks = JapaneseMark.Matches(text).Cast<Match>().Select(m => m.Value).ToList();
        var phones = new List<string>();
        for (int i = 0; i < sentences.Length; i++)
        {
            if (JapaneseCharacter.IsMatch(sentences[i]))
            {
                var prosody = Prosody(sentences[i]);
                phones.AddRange(prosody.Skip(1).Take(Math.Max(0, prosody.Count - 2)));
            }
            if (i < marks.Count)
            {
                if (marks[i] == " ") continue;
                phones.Add(marks[i].Replace(" ", ""));
            }
        }
        return phones.Select(PostReplace).ToList();
    }

    private static string PostReplace(string ph) => ph switch
    {
        "\uff1a" or "\uff1b" or "\uff0c" or "\u00b7" or "\u3001" => ",",
        "\u3002" or "\n" => ".",
        "\uff01" => "!",
        "\uff1f" => "?",
        "..." => "\u2026",
        _ => ph,
    };

    private static readonly Regex CurrentPhoneme = new(@"\-(.*?)\+");
    private static readonly Regex A1 = new(@"/A:([0-9\-]+)\+");
    private static readonly Regex A2 = new(@"\+(\d+)\+");
    private static readonly Regex A3 = new(@"\+(\d+)/");
    private static readonly Regex F1 = new(@"/F:(\d+)_");
    private static readonly Regex E3 = new(@"!(\d+)_");

    private static int Numeric(Regex regex, string s)
    {
        var match = regex.Match(s);
        return match.Success ? int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) : -50;
    }

    /// <summary><c>pyopenjtalk_g2p_prosody(text)</c>: phonemes and prosody marks between "^" and "$" (or "?").</summary>
    internal List<string> Prosody(string text)
    {
        var labels = OpenJTalk.MakeLabel(_openJTalk.RunFrontend(text));
        var phones = new List<string>();
        for (int n = 0; n < labels.Count; n++)
        {
            string label = labels[n];
            string p3 = CurrentPhoneme.Match(label).Groups[1].Value;
            if ("AEIOU".Contains(p3)) p3 = p3.ToLowerInvariant();
            if (p3 == "sil")
            {
                if (n != 0 && n != labels.Count - 1) throw new InvalidOperationException("A silence inside the utterance.");
                if (n == 0) phones.Add("^");
                else
                {
                    int e3 = Numeric(E3, label);
                    if (e3 == 0) phones.Add("$");
                    else if (e3 == 1) phones.Add("?");
                }
                continue;
            }
            if (p3 == "pau")
            {
                phones.Add("_");
                continue;
            }
            phones.Add(p3);
            int a1 = Numeric(A1, label), a2 = Numeric(A2, label), a3 = Numeric(A3, label), f1 = Numeric(F1, label);
            int a2Next = Numeric(A2, labels[n + 1]);
            if (a3 == 1 && a2Next == 1 && "aeiouAEIOUNcl".Contains(p3)) phones.Add("#");
            else if (a1 == 0 && a2Next == a2 + 1 && a2 != f1) phones.Add("]");
            else if (a2 == 1 && a2Next == 2) phones.Add("[");
        }
        return phones;
    }
}
