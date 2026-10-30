using System.Globalization;
using System.Text;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpiritLM's speech-token strings (the release's <c>spiritlm_tokenizer.py</c>): <c>[Hu…]</c> HuBERT units at 24.99 a
/// second, <c>[Pi…]</c> pitch tokens at 12.5 and <c>[St…]</c> style tokens at 1, each deduplicated as the release does
/// (HuBERT and pitch merged when repeated, style not) and interleaved by time.
/// </summary>
internal static class SpiritLmTokens
{
    public const string TextPrefix = "[Text]", SpeechPrefix = "[Speech]";

    /// <summary>The release's fallback styles when pitch tokens come without any style token.</summary>
    public static readonly int[] MostCommonStyles = { 71, 68, 98 };

    /// <summary>The rates and deduplication <c>units_to_string</c> uses.</summary>
    internal sealed record Rates(double Hubert = 24.99, double Pitch = 12.5, double Style = 1.0, bool DeduplicateHubert = true,
        bool DeduplicatePitch = true, bool DeduplicateStyle = false);

    /// <summary>
    /// <c>units_to_string</c>: each stream's tokens at <c>i / rate</c> seconds (repeats dropped where deduplicated), merged
    /// by time with a stable sort in the order style, pitch, HuBERT.
    /// </summary>
    public static string UnitsToString(IReadOnlyList<int> hubert, IReadOnlyList<int>? pitch = null, IReadOnlyList<int>? style = null, Rates? rates = null)
    {
        rates ??= new Rates();
        var tokens = new List<(string Token, double Time)>();
        void Add(string prefix, IReadOnlyList<int> values, double rate, bool deduplicate)
        {
            for (int i = 0; i < values.Count; i++)
                if (!deduplicate || i == 0 || values[i] != values[i - 1])
                    tokens.Add(($"[{prefix}{values[i].ToString(CultureInfo.InvariantCulture)}]", i / rate));
        }
        if (style is not null) Add("St", style, rates.Style, rates.DeduplicateStyle);
        if (pitch is not null) Add("Pi", pitch, rates.Pitch, rates.DeduplicatePitch);
        Add("Hu", hubert, rates.Hubert, rates.DeduplicateHubert);
        var sb = new StringBuilder();
        foreach (var token in tokens.Select((t, i) => (t, i)).OrderBy(x => x.t.Time).ThenBy(x => x.i)) sb.Append(token.t.Token);
        return sb.ToString();
    }

    /// <summary>
    /// <c>string_to_units</c>: the HuBERT units of a token string, each with the pitch and style tokens last seen before it
    /// (those before the first HuBERT unit apply to it; units before any pitch or style take the first); pitch without any
    /// style takes one of the release's most common styles, drawn by <paramref name="random"/>. Text and other bracketed
    /// pieces are ignored.
    /// </summary>
    public static (int[] Hubert, int[]? Pitch, int[]? Style) StringToUnits(string generated, Random? random = null, bool duplicateHubertForMultiplePitch = false)
    {
        int? previousPitch = null, firstPitch = null, previousStyle = null, firstStyle = null, previousHubert = null, firstHubert = null;
        bool previousIsPitch = false;
        var hubert = new List<int?>();
        var pitch = new List<int?>();
        var style = new List<int?>();
        foreach (var item in generated.Split('['))
        {
            if (item.Length <= 2) continue;
            // item[2:-1]: the digits after the two-letter prefix, the closing bracket dropped as Python slices it.
            bool digit = char.IsDigit(item[2]);
            int Value() => int.Parse(item.Substring(2, item.Length - 3), CultureInfo.InvariantCulture);
            if (item.StartsWith("Hu", StringComparison.Ordinal) && digit)
            {
                int v = Value();
                hubert.Add(v);
                pitch.Add(previousPitch);
                style.Add(previousStyle);
                previousIsPitch = false;
                previousHubert = v;
                firstHubert ??= v;
            }
            else if (item.StartsWith("St", StringComparison.Ordinal) && digit)
            {
                int v = Value();
                if (previousStyle is null) firstStyle = v;
                previousStyle = v;
            }
            else if (item.StartsWith("Pi", StringComparison.Ordinal) && digit)
            {
                int v = Value();
                if (duplicateHubertForMultiplePitch && previousIsPitch)
                {
                    hubert.Add(previousHubert);
                    pitch.Add(v);
                    style.Add(previousStyle);
                }
                if (previousPitch is null) firstPitch = v;
                previousPitch = v;
                previousIsPitch = true;
            }
        }
        if (firstPitch is not null && firstStyle is null)
            firstStyle = MostCommonStyles[(random ?? new Random()).Next(MostCommonStyles.Length)];
        for (int i = 0; i < hubert.Count; i++)
        {
            hubert[i] ??= firstHubert;
            style[i] ??= firstStyle;
            pitch[i] ??= firstPitch;
        }
        var hubertOut = hubert.Select(v => v!.Value).ToArray();
        return (hubertOut, firstPitch is null ? null : pitch.Select(v => v!.Value).ToArray(), firstStyle is null ? null : style.Select(v => v!.Value).ToArray());
    }

    private static bool IsSpeechTag(string s, int at) =>
        at + 2 <= s.Length && (string.CompareOrdinal(s, at, "Hu", 0, 2) == 0 || string.CompareOrdinal(s, at, "Pi", 0, 2) == 0 || string.CompareOrdinal(s, at, "St", 0, 2) == 0);

    /// <summary><c>does_start_with_speech_token</c>: the string opens with <c>[Hu…]</c>, <c>[Pi…]</c> or <c>[St…]</c> (its
    /// closing bracket within ten characters).</summary>
    public static bool StartsWithSpeechToken(string? s)
    {
        if (s is null || s.Length <= 4 || s[0] != '[') return false;
        for (int end = 1; end < s.Length; end++)
        {
            if (s[end] == ']' && end >= 4) return IsSpeechTag(s, 1);
            if (end >= 10) return false;
        }
        return false;
    }

    /// <summary><c>does_end_with_speech_token</c>: the string closes with a speech token (its opening bracket within ten
    /// characters of the end).</summary>
    public static bool EndsWithSpeechToken(string? s)
    {
        if (s is null || s.Length <= 4 || s[s.Length - 1] != ']') return false;
        for (int start = s.Length - 2; start >= 0; start--)
        {
            if (s[start] == '[' && start + 3 < s.Length) return IsSpeechTag(s, start + 1);
            if (start < s.Length - 10) return false;
        }
        return false;
    }

    /// <summary><c>find_prompt_last_speech_start_position</c>: where the prompt's last unbroken run of speech tokens
    /// starts (null without one).</summary>
    public static int? LastSpeechStart(string prompt)
    {
        var reversed = new string(prompt.Reverse().ToArray());
        int? previousEnd = null;
        foreach (System.Text.RegularExpressions.Match match in System.Text.RegularExpressions.Regex.Matches(reversed, @"(\]\d+uH\[)|(\]\d+iP\[)|(\]\d+tS\[)"))
        {
            int start = match.Index, end = match.Index + match.Length;
            if (previousEnd is not null && start != previousEnd) return prompt.Length - previousEnd.Value;
            previousEnd = end;
        }
        return previousEnd is null ? null : prompt.Length - previousEnd.Value;
    }

    /// <summary>
    /// <c>_parse_speech_and_text</c>: generated content split into runs of text and of speech tokens, the
    /// <c>[Text]</c> and <c>[Speech]</c> markers dropped (other bracketed text is text).
    /// </summary>
    public static List<(string Content, bool IsSpeech)> SplitSpeechAndText(string content)
    {
        var splits = new List<(string, bool)>();
        var current = new StringBuilder();
        bool? currentIsSpeech = null;
        int i = 0, last = content.Length;
        while (i < last)
        {
            int j = i;
            bool speech = false, text = false;
            if (content[i] == '[')
            {
                if (j + TextPrefix.Length - 1 < last && string.CompareOrdinal(content, j, TextPrefix, 0, TextPrefix.Length) == 0) j += TextPrefix.Length;
                else if (j + SpeechPrefix.Length - 1 < last && string.CompareOrdinal(content, j, SpeechPrefix, 0, SpeechPrefix.Length) == 0) j += SpeechPrefix.Length;
                else if (j + 2 < last && IsSpeechTag(content, j + 1))
                {
                    j += 3;
                    while (j < last && content[j] != ']') j++;
                    j++;
                    speech = true;
                }
                else
                {
                    text = true;
                    j++;
                }
            }
            else
            {
                text = true;
                while (j < last && content[j] != '[') j++;
            }
            j = Math.Min(j, last);
            if (speech || text)
            {
                if (currentIsSpeech is not null && currentIsSpeech != speech)
                {
                    splits.Add((current.ToString(), currentIsSpeech.Value));
                    current.Clear();
                }
                current.Append(content, i, j - i);
                currentIsSpeech = speech;
            }
            i = j;
        }
        if (currentIsSpeech is not null && current.Length > 0) splits.Add((current.ToString(), currentIsSpeech.Value));
        return splits;
    }

    /// <summary>The token ids the release forbids (<c>get_forbidden_tokens</c>): the modality markers, and every text token
    /// when generating speech, or every speech token (with the expressive ones when <paramref name="expressive"/>) when
    /// generating text.</summary>
    public static HashSet<int> ForbiddenTokens(bool onlySpeech, bool onlyText, bool expressive, int textVocabulary = 32000, int hubertUnits = 501,
        int pitchTokens = 64, int styleTokens = 100)
    {
        if (onlySpeech && onlyText) throw new ArgumentException("Nothing can be generated when only speech and only text are both asked for.");
        var forbidden = new HashSet<int> { textVocabulary, textVocabulary + 1 };
        if (onlySpeech) for (int i = 0; i < textVocabulary; i++) forbidden.Add(i);
        else if (onlyText)
        {
            int hu = textVocabulary + 2;
            for (int i = 0; i < hubertUnits; i++) forbidden.Add(hu + i);
            if (expressive)
                for (int i = 0; i < pitchTokens + styleTokens; i++) forbidden.Add(hu + hubertUnits + i);
        }
        return forbidden;
    }
}
