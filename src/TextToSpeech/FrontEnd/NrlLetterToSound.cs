using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The letter-to-sound rules of NRL Report 7948 (Elovitz, Johnson, McHugh and Shore 1976, "Automatic Translation of
/// English Text to Phonetics by Means of Letter-to-Sound Rules"), applied as the report's TRANS program applies them.
/// </summary>
/// <remarks>
/// <para>
/// Each rule reads <c>left[match]right = phones</c>. For the letter at the cursor the rules of that letter's group are
/// tried in order; the first whose bracketed letters match at the cursor, whose left context matches the text just
/// before it and whose right context matches the text just after it emits its phones, and the cursor moves past the
/// bracketed letters. Words are delimited by blanks, which the contexts can name.
/// </para>
/// <para>
/// Context symbols (TRANS's patterns): <c>#</c> one or more vowels (AEIOUY), <c>*</c> one or more consonants,
/// <c>.</c> a voiced consonant (BDVGJLMNRWZ), <c>$</c> a consonant then E or I, <c>%</c> one of the suffixes ER, E, ES,
/// ED or ING followed by a blank, <c>&amp;</c> a sibilant (S C G Z X J, CH, SH), <c>@</c> a consonant after which long
/// U is pronounced as in "rule" (T S R D L Z N J, TH, CH, SH), <c>^</c> one consonant, <c>+</c> a front vowel (E I Y)
/// and <c>:</c> zero or more consonants. The 329 rules are an embedded resource extracted from the report's SNOBOL4
/// program by <c>tools/reference-data/g2p_resources.py</c>.
/// </para>
/// <para>Phones are the report's ARPAbet-like set (AX is the schwa, NX the velar nasal). Vowels come back with a
/// primary stress mark, AX with none, so they read like the CMU dictionary's entries.</para>
/// </remarks>
internal static class NrlLetterToSound
{
    private const string ResourceName = "AiDotNet.TextToSpeech.FrontEnd.nrl_rules.tsv";
    private const string Consonant = "[BCDFGHJKLMNPQRSTVWXZ]";

    private static readonly System.Lazy<Dictionary<string, List<Rule>>> Rules = new(Load);

    private static readonly HashSet<string> Vowels = new(System.StringComparer.Ordinal)
    {
        "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    };

    private sealed class Rule
    {
        public Rule(Regex left, string match, Regex right, string[] phones)
        {
            Left = left;
            Match = match;
            Right = right;
            Phones = phones;
        }

        public Regex Left { get; }
        public string Match { get; }
        public Regex Right { get; }
        public string[] Phones { get; }
    }

    /// <summary>The ARPAbet phones the rules give for one word.</summary>
    public static string[] Translate(string word)
    {
        string text = " " + word.ToUpperInvariant() + " ";
        var rules = Rules.Value;
        var phones = new List<string>();
        int i = 1;
        while (i < text.Length - 1)
        {
            char ch = text[i];
            string group = char.IsDigit(ch) ? "NUMBER" : char.IsLetter(ch) ? ch.ToString() : "PUNCT";
            bool applied = false;
            if (rules.TryGetValue(group, out var candidates))
            {
                foreach (var rule in candidates)
                {
                    if (i + rule.Match.Length > text.Length
                        || string.CompareOrdinal(text, i, rule.Match, 0, rule.Match.Length) != 0) continue;
                    if (!rule.Left.IsMatch(text.Substring(0, i))) continue;
                    if (!rule.Right.Match(text, i + rule.Match.Length).Success) continue;
                    foreach (var phone in rule.Phones)
                        if (phone.Length > 0 && char.IsLetter(phone[0])) phones.Add(phone);
                    i += rule.Match.Length;
                    applied = true;
                    break;
                }
            }
            if (!applied) i++;
        }
        for (int k = 0; k < phones.Count; k++)
            if (Vowels.Contains(phones[k])) phones[k] += phones[k] == "AX" ? "0" : "1";
        return phones.ToArray();
    }

    private static string ContextPattern(string context)
    {
        var pattern = new StringBuilder();
        foreach (char c in context)
        {
            pattern.Append(c switch
            {
                '#' => "[AEIOUY]+",
                '*' => Consonant + "+",
                '.' => "[BDVGJLMNRWZ]",
                '$' => Consonant + "[EI]",
                '%' => "(?:ER |E |ES |ED |ING )",
                '&' => "(?:[SCGZXJ]|CH|SH)",
                '@' => "(?:[TSRDLZNJ]|TH|CH|SH)",
                '^' => Consonant,
                '+' => "[EIY]",
                ':' => Consonant + "*",
                _ => Regex.Escape(c.ToString()),
            });
        }
        return pattern.ToString();
    }

    private static Dictionary<string, List<Rule>> Load()
    {
        var assembly = typeof(NrlLetterToSound).Assembly;
        using var stream = assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidDataException($"The embedded resource {ResourceName} is missing.");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        var rules = new Dictionary<string, List<Rule>>(System.StringComparer.Ordinal);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0) continue;
            var fields = line.Split('\t');
            if (fields.Length != 5) throw new InvalidDataException($"Malformed NRL rule: '{line}'.");
            var options = RegexOptions.CultureInvariant;
            var rule = new Rule(
                new Regex("(?:" + ContextPattern(fields[1]) + ")$", options),
                fields[2],
                new Regex(@"\G(?:" + ContextPattern(fields[3]) + ")", options),
                fields[4].Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries));
            if (!rules.TryGetValue(fields[0], out var group)) rules[fields[0]] = group = new List<Rule>();
            group.Add(rule);
        }
        return rules;
    }
}
