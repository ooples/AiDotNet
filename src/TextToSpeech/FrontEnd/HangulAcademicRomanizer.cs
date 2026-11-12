using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>hangul_romanize.rule.academic, including boundary markers for ambiguous consonant sequences.</summary>
internal static class HangulAcademicRomanizer
{
    private static readonly string[] Initials = ["g", "kk", "n", "d", "tt", "l", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"];
    private static readonly string[] Vowels = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"];
    private static readonly string[] Finals = ["", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"];
    private static readonly HashSet<string> Ambiguous = BuildAmbiguous();

    private static HashSet<string> BuildAmbiguous()
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        foreach (string final in Finals) foreach (string initial in Initials)
        {
            string both = final + initial;
            bool found = false;
            for (int i = 0; i < both.Length; i++)
                if (Finals.Contains(both.Substring(0, i)) && Initials.Contains(both.Substring(i)))
                {
                    if (found) { result.Add(both); break; }
                    found = true;
                }
        }
        return result;
    }

    internal static string Romanize(string text)
    {
        var result = new StringBuilder(); int previous = -1;
        foreach (char c in text)
        {
            if (c < 0xAC00 || c > 0xD7A3) { result.Append(c); previous = -1; continue; }
            int syllable = c - 0xAC00, initial = syllable / 588;
            if (previous >= 0 && (initial == 11 || Ambiguous.Contains(Finals[previous % 28] + Initials[initial]))) result.Append('-');
            result.Append(Initials[initial]).Append(Vowels[(syllable / 28) % 21]).Append(Finals[syllable % 28]);
            previous = syllable;
        }
        return result.ToString();
    }
}
