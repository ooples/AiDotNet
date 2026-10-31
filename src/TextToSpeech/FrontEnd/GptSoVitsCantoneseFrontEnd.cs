using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// ToJyutping 3.2.0 (CanCLID; BSD-2-Clause): each character's Jyutping from the longest dictionary word starting at it
/// (<c>get_jyutping_list</c>), over its trie exported by <c>tools/reference-data/gptsovits_korean_cantonese_reference.py</c>.
/// </summary>
internal sealed class ToJyutping
{
    private static readonly Lazy<ToJyutping> SharedInstance = new(() => new ToJyutping(), isThreadSafe: true);
    private readonly Dictionary<string, string[]> _words = new(StringComparer.Ordinal);
    private readonly HashSet<string> _prefixes = new(StringComparer.Ordinal);
    private int _maxLength;

    public static ToJyutping Default => SharedInstance.Value;

    private ToJyutping()
    {
        foreach (var line in ChineseText.ReadLines("tojyutping_words.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            string word = line.Substring(0, tab);
            _words[word] = line.Substring(tab + 1).Split(' ');
            var codepoints = ChineseText.Codepoints(word);
            _maxLength = Math.Max(_maxLength, codepoints.Count);
            var prefix = new StringBuilder();
            foreach (var c in codepoints)
            {
                prefix.Append(c);
                _prefixes.Add(prefix.ToString());
            }
        }
    }

    /// <summary><c>get_jyutping_list(text)</c>: each character with its syllable, or null where there is none.</summary>
    public List<(string Char, string? Jyutping)> Read(string text)
    {
        var chars = ChineseText.Codepoints(text);
        var result = new List<(string, string?)>();
        int i = 0;
        while (i < chars.Count)
        {
            string[]? best = null;
            int k = i;
            var key = new StringBuilder();
            for (int j = i; j < chars.Count; j++)
            {
                key.Append(chars[j]);
                string candidate = key.ToString();
                if (!_prefixes.Contains(candidate)) break;
                if (_words.TryGetValue(candidate, out var value))
                {
                    best = value;
                    k = j;
                }
            }
            if (k == i)
            {
                result.Add((chars[i], best?[0]));
                i++;
            }
            else
            {
                for (int n = i; i <= k; i++) result.Add((chars[i], best![i - n]));
            }
        }
        return result;
    }
}

/// <summary>
/// GPT-SoVITS's Cantonese front end (RVC-Boss/GPT-SoVITS at b4570b8, MIT; <c>text/cantonese.py</c>, after
/// Naozumi0512's Bert-VITS2-Cantonese-Yue): zh_normalization, its punctuation map with everything outside Chinese and the
/// six punctuation marks dropped; then ToJyutping's syllables split into an initial and a toned final, each prefixed "Y".
/// </summary>
internal static class GptSoVitsCantoneseFrontEnd
{
    private static readonly (string From, string To)[] Replacements =
    {
        ("\uff1a", ","), ("\uff1b", ","), ("\uff0c", ","), ("\u3002", "."), ("\uff01", "!"), ("\uff1f", "?"), ("\n", "."), ("\u00b7", ","),
        ("\u3001", ","), ("...", "\u2026"), ("$", "."), ("\u201c", "'"), ("\u201d", "'"), ("\"", "'"), ("\u2018", "'"), ("\u2019", "'"),
        ("\uff08", "'"), ("\uff09", "'"), ("(", "'"), (")", "'"), ("\u300a", "'"), ("\u300b", "'"), ("\u3010", "'"), ("\u3011", "'"),
        ("[", "'"), ("]", "'"), ("\u2014", "-"), ("\uff5e", "-"), ("~", "-"), ("\u300c", "'"), ("\u300d", "'"),
    };

    private static readonly Regex ReplacementPattern = new(string.Join("|", Replacements.Select(r => Regex.Escape(r.From))));
    private static readonly Regex Outside = new("[^\u4e00-\u9fa5!?\u2026,.\\-]+");

    /// <summary><c>cantonese.text_normalize</c>.</summary>
    public static string TextNormalize(string text)
        => string.Concat(GptSoVitsChineseNormalizer.Normalize(text).Select(sentence =>
            Outside.Replace(ReplacementPattern.Replace(sentence, m => Replacements.First(r => r.From == m.Value).To), "")));

    private static readonly string[] Initials =
    {
        "aa", "aai", "aak", "aap", "aat", "aau", "ai", "au", "ap", "at", "ak", "a", "p", "b", "e", "ts", "t", "dz", "d", "kw", "k", "gw", "g",
        "f", "h", "l", "m", "ng", "n", "s", "y", "w", "c", "z", "j", "ong", "on", "ou", "oi", "ok", "o", "uk", "ung", "sp", "spl", "spn", "sil",
    };

    private static readonly Regex Syllables = new("^([a-z]+[1-6]+[ ]?)+$");

    // get_jyutping: punctuation kept, every other character's syllable required.
    private static List<string> Jyutping(string text)
    {
        var result = new List<string>();
        foreach (var (word, syllable) in ToJyutping.Default.Read(text))
        {
            if (word.Length > 0 && word.All(c => GptSoVitsChineseFrontEnd.Punctuation.Contains(c.ToString())))
            {
                result.Add(word);
                continue;
            }
            // re.search on a missing syllable raises TypeError in the reference.
            if (syllable is null) throw new ArgumentException($"ToJyutping has no reading for '{word}'.", nameof(text));
            if (!Syllables.IsMatch(syllable)) throw new ArgumentException($"Failed to convert {word} to jyutping: {syllable}", nameof(text));
            result.Add(syllable);
        }
        return result;
    }

    /// <summary><c>cantonese.g2p(text)</c> of normalized text: the phones and the phones per character.</summary>
    public static (List<string> Phones, List<int> WordToPhones) G2P(string text)
    {
        var parts = new List<(string Part, int Tone)>();
        var wordToPhones = new List<int>();
        foreach (var syllable in Jyutping(text))
        {
            if (GptSoVitsChineseFrontEnd.Punctuation.Contains(syllable) || syllable == "_")
            {
                parts.Add((syllable, 0));
                wordToPhones.Add(1);
                continue;
            }
            char last = syllable[syllable.Length - 1];
            int tone = last is >= '0' and <= '9' ? last - '0' : 0;
            string bare = last is >= '0' and <= '9' ? syllable.Substring(0, syllable.Length - 1) : syllable;
            foreach (var initial in Initials)
            {
                if (!bare.StartsWith(initial, StringComparison.Ordinal)) continue;
                if (bare.StartsWith("nga", StringComparison.Ordinal))
                {
                    string rest = bare.Substring(2);
                    parts.Add((bare.Substring(0, 2), -1));
                    parts.Add((rest.Length > 0 ? rest : bare.Substring(bare.Length - 1), tone));
                }
                else
                {
                    string final = bare.Substring(initial.Length);
                    parts.Add((initial, -1));
                    parts.Add((final.Length > 0 ? final : initial.Substring(initial.Length - 1), tone));
                }
                wordToPhones.Add(2);
                break;
            }
        }
        var phones = new List<string>();
        foreach (var (part, tone) in parts)
        {
            string todo = tone is -1 or 0 ? part : part + tone;
            phones.Add(GptSoVitsChineseFrontEnd.Punctuation.Contains(todo) ? todo : "Y" + todo);
        }
        return (phones, wordToPhones);
    }
}
