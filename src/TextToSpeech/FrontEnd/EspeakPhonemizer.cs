using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Phonemization through eSpeak NG with the behavior of the phonemizer package's espeak backend (version 3.3, with
/// punctuation preserved, stress removed, language-switch flags removed and the "strip" option), which TTS front ends such
/// as MaskGCT's call. Written from that behavior, not from phonemizer's source.
/// </summary>
/// <remarks>
/// <para>A line is cut at each run of punctuation (optional whitespace, then one or more of the marks, then optional
/// whitespace, repeated), the pieces between runs are phonemized one by one, and the runs are put back where they were,
/// their spaces turned into the word separator. eSpeak's output for a piece is cleaned as phonemizer cleans it: clauses
/// merged onto one line, repeated '_' collapsed, a '_' before a space dropped, flags such as "(en)" removed, stress marks
/// (ˈ, ˌ, ', -) removed from each word, the in-word '_' replaced by the phone separator, and the words joined by the word
/// separator.</para>
/// <para>The eSpeak call is a function of the language and the text, so tests can replay recorded output; production uses
/// <see cref="EspeakNg.TextToPhonemes"/>.</para>
/// </remarks>
internal sealed class EspeakPhonemizer
{
    private static readonly Regex StressMarks = new("[ˈˌ'-]+", RegexOptions.CultureInvariant);
    private static readonly Regex LanguageFlags = new(@"\(.+?\)", RegexOptions.CultureInvariant);
    private static readonly Regex RepeatedUnderscores = new("_+", RegexOptions.CultureInvariant);

    private readonly string _language;
    private readonly Func<string, string, string> _espeak;
    private readonly Regex _punctuation;

    /// <param name="language">eSpeak's language code ("en-us", "fr-fr", "de", "ko", …).</param>
    /// <param name="marks">The punctuation marks kept (each one character).</param>
    /// <param name="espeak">eSpeak's phonemes of (language, text), clauses joined by spaces; null for the installed
    /// library.</param>
    public EspeakPhonemizer(string language, string marks, Func<string, string, string>? espeak = null)
    {
        _language = language ?? throw new ArgumentNullException(nameof(language));
        _espeak = espeak ?? ((lang, text) => EspeakNg.Instance.TextToPhonemes(lang, text));
        var set = new StringBuilder();
        foreach (char c in marks.Distinct())
        {
            if (c == '\\' || c == ']' || c == '[' || c == '^' || c == '-') set.Append('\\');
            set.Append(c);
        }
        _punctuation = new Regex(@"(\s*[" + set + @"]+\s*)+", RegexOptions.CultureInvariant);
    }

    /// <summary>The phonemes of one line with the phone and word separators given, punctuation kept in place.</summary>
    public string Phonemize(string line, string phoneSeparator, string wordSeparator)
    {
        // The punctuation runs, each with where it sits: the whole line, its start, its end, or inside.
        var runs = _punctuation.Matches(line).Cast<Match>().Where(m => m.Length > 0).ToList();
        if (runs.Count == 0) return Clean(_espeak(_language, line), phoneSeparator, wordSeparator);
        if (runs.Count == 1 && runs[0].Value == line) return line.Replace(" ", wordSeparator);

        // Cut the line at each run's text (its first remaining occurrence), keeping the non-empty pieces.
        var pieces = new List<string>();
        string rest = line;
        foreach (var run in runs)
        {
            int at = rest.IndexOf(run.Value, StringComparison.Ordinal);
            pieces.Add(rest.Substring(0, at));
            rest = rest.Substring(at + run.Value.Length);
        }
        pieces.Add(rest);
        var text = pieces.Where(p => p.Length > 0).Select(p => Clean(_espeak(_language, p), phoneSeparator, wordSeparator)).ToList();

        // Put the runs back, in order: a leading run goes in front of the current piece, an inner run joins the current
        // piece to the next, a trailing run ends the current piece; a piece's trailing word separator is dropped when a run
        // is attached to it.
        var output = new List<string>();
        for (int i = 0; i < runs.Count; i++)
        {
            string mark = runs[i].Value.Replace(" ", wordSeparator);
            if (text.Count == 0)
            {
                output.Add(string.Concat(runs.Skip(i).Select(r => r.Value)).Replace(" ", wordSeparator));
                break;
            }
            if (wordSeparator.Length > 0 && text[0].EndsWith(wordSeparator, StringComparison.Ordinal))
                text[0] = text[0].Substring(0, text[0].Length - wordSeparator.Length);
            bool leading = i == 0 && line.StartsWith(runs[i].Value, StringComparison.Ordinal);
            bool trailing = !leading && i == runs.Count - 1 && line.EndsWith(runs[i].Value, StringComparison.Ordinal);
            if (leading)
            {
                text[0] = mark + text[0];
            }
            else if (trailing)
            {
                output.Add(text[0] + mark);
                text.RemoveAt(0);
            }
            else if (text.Count == 1)
            {
                text[0] += mark;
            }
            else
            {
                text[1] = text[0] + mark + text[1];
                text.RemoveAt(0);
            }
        }
        output.AddRange(text);
        return string.Join("\n", output);
    }

    // eSpeak's output for one piece, cleaned and separated (phonemizer's line post-processing with strip).
    private static string Clean(string espeak, string phoneSeparator, string wordSeparator)
    {
        string line = espeak.Trim().Replace("\n", " ").Replace("  ", " ");
        line = RepeatedUnderscores.Replace(line, "_");
        line = line.Replace("_ ", " ");
        line = LanguageFlags.Replace(line, "");
        if (line.Length == 0) return "";
        return string.Join(wordSeparator, line.Split(' ').Select(word => StressMarks.Replace(word.Trim(), "").Replace("_", phoneSeparator)));
    }
}
