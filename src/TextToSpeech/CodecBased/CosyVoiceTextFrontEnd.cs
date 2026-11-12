using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// CosyVoice's text front end without an external normalizer (FunAudioLLM/CosyVoice <c>cli/frontend.py</c>
/// <c>text_normalize</c> with <c>text_frontend == ''</c>, and <c>utils/frontend_utils.py</c>): Chinese text has the blanks
/// between non-ASCII characters removed, ² and ³ read out, '.' made '。', " - " made '，', brackets removed and a trailing
/// comma made '。'; other text has its digits spelled out by <c>inflect</c>. Both are then split into sentences of about
/// 60–80 units (characters for Chinese, tokens otherwise), a short last one merged into the one before, and sentences of
/// punctuation alone dropped.
/// </summary>
/// <remarks>
/// Text holding a special token (<c>&lt;|…|&gt;</c>) passes through whole, as the reference skips its front end then.
/// The release can instead run the <c>wetext</c> or <c>ttsfrd</c> WFST normalizers, which are separate packages; text
/// normalized by one of them can be passed in directly.
/// </remarks>
internal static class CosyVoiceTextFrontEnd
{
    private static readonly Regex TrailingComma = new("[，,、]+$", RegexOptions.CultureInvariant);
    private static readonly char[] ChinesePunctuation = { '。', '？', '！', '；', '：', '、', '.', '?', '!', ';' };
    private static readonly char[] OtherPunctuation = { '.', '?', '!', ';', ':' };

    /// <summary>The sentences of <paramref name="text"/> (<c>text_normalize(text, split=True)</c>).</summary>
    /// <param name="text">The text.</param>
    /// <param name="tokenCount">The number of text tokens a string encodes to (English sentence lengths count them).</param>
    public static IReadOnlyList<string> Sentences(string text, Func<string, int> tokenCount)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (tokenCount is null) throw new ArgumentNullException(nameof(tokenCount));
        if (SkipsFrontEnd(text)) return new[] { text };
        var (normalized, chinese) = Rules(text);
        return SplitParagraph(normalized, tokenCount, chinese).Where(sentence => !IsOnlyPunctuation(sentence)).ToList();
    }

    /// <summary>The normalized text, unsplit (<c>text_normalize(text, split=False)</c>, as a prompt's transcript is).</summary>
    public static string Normalize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return SkipsFrontEnd(text) ? text : Rules(text).Text;
    }

    private static bool SkipsFrontEnd(string text) => text.Length == 0 || (text.Contains("<|") && text.Contains("|>"));

    private static (string Text, bool Chinese) Rules(string text)
    {
        text = PythonStrip(text);
        if (ContainsChinese(text))
        {
            text = text.Replace("\n", "");
            text = ReplaceBlank(text);
            text = text.Replace("²", "平方").Replace("³", "立方");
            text = text.Replace(".", "。").Replace(" - ", "，");
            text = text.Replace("（", "").Replace("）", "").Replace("【", "").Replace("】", "").Replace("`", "").Replace("——", " ");
            return (TrailingComma.Replace(text, "。"), true);
        }
        return (SpellOutNumber(text), false);
    }

    internal static bool ContainsChinese(string text) => text.Any(c => c >= '一' && c <= '鿿');

    // str.strip(): Python's whitespace adds the four information separators U+001C–U+001F to .NET's.
    private static bool IsPythonSpace(char c) => char.IsWhiteSpace(c) || (c >= '\u001c' && c <= '\u001f');

    private static string PythonStrip(string text)
    {
        int start = 0, end = text.Length;
        while (start < end && IsPythonSpace(text[start])) start++;
        while (end > start && IsPythonSpace(text[end - 1])) end--;
        return text.Substring(start, end - start);
    }

    // replace_blank: a space survives only between two non-space ASCII characters.
    internal static string ReplaceBlank(string text)
    {
        var output = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c != ' ')
            {
                output.Append(c);
                continue;
            }
            // text[i - 1] at 0 is the last character in Python; the stripped text never starts or ends with a space.
            char before = text[i == 0 ? text.Length - 1 : i - 1];
            if (i + 1 < text.Length && text[i + 1] < 128 && text[i + 1] != ' ' && before < 128 && before != ' ')
                output.Append(c);
        }
        return output.ToString();
    }

    // spell_out_number: each run of str.isdigit() characters through inflect's number_to_words.
    internal static string SpellOutNumber(string text)
    {
        var output = new StringBuilder(text.Length);
        int start = -1;
        for (int i = 0; i < text.Length;)
        {
            // Python iterates code points: a supplementary digit (𝟙) is one character.
            int width = char.IsSurrogatePair(text, i) ? 2 : 1;
            if (CharUnicodeInfo.GetDigitValue(text, i) >= 0)
            {
                if (start < 0) start = i;
            }
            else
            {
                if (start >= 0)
                {
                    output.Append(InflectNumberWords.NumberToWords(text.Substring(start, i - start)));
                    start = -1;
                }
                output.Append(text, i, width);
            }
            i += width;
        }
        if (start >= 0) output.Append(InflectNumberWords.NumberToWords(text.Substring(start)));
        return output.ToString();
    }

    // split_paragraph(text, tokenize, lang, token_max_n=80, token_min_n=60, merge_len=20, comma_split=False).
    internal static List<string> SplitParagraph(string text, Func<string, int> tokenCount, bool chinese)
    {
        const int maxLength = 80, minLength = 60, mergeLength = 20;
        int Length(string s) => chinese ? CodePoints(s) : tokenCount(s);
        var punctuation = chinese ? ChinesePunctuation : OtherPunctuation;
        if (text.Length == 0) return new List<string>();
        if (Array.IndexOf(punctuation, text[text.Length - 1]) < 0) text += chinese ? "。" : ".";

        var sentences = new List<string>();
        int st = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (Array.IndexOf(punctuation, text[i]) < 0) continue;
            if (i - st > 0) sentences.Add(text.Substring(st, i - st) + text[i]);
            if (i + 1 < text.Length && (text[i + 1] == '"' || text[i + 1] == '”'))
            {
                // The reference pops the last sentence even when none was added for this mark.
                if (sentences.Count == 0) throw new InvalidOperationException("A closing quote follows punctuation that starts the text.");
                var last = sentences[sentences.Count - 1];
                sentences[sentences.Count - 1] = last + text[i + 1];
                st = i + 2;
            }
            else
            {
                st = i + 1;
            }
        }

        var merged = new List<string>();
        string current = "";
        foreach (var sentence in sentences)
        {
            if (Length(current + sentence) > maxLength && Length(current) > minLength)
            {
                merged.Add(current);
                current = "";
            }
            current += sentence;
        }
        if (current.Length > 0)
        {
            if (Length(current) < mergeLength && merged.Count != 0) merged[merged.Count - 1] += current;
            else merged.Add(current);
        }
        return merged;
    }

    // is_only_punctuation: every character punctuation (\p{P}) or a symbol (\p{S}); the empty string too.
    internal static bool IsOnlyPunctuation(string text)
    {
        for (int i = 0; i < text.Length; i += char.IsSurrogatePair(text, i) ? 2 : 1)
        {
            switch (CharUnicodeInfo.GetUnicodeCategory(text, i))
            {
                case UnicodeCategory.ConnectorPunctuation:
                case UnicodeCategory.DashPunctuation:
                case UnicodeCategory.OpenPunctuation:
                case UnicodeCategory.ClosePunctuation:
                case UnicodeCategory.InitialQuotePunctuation:
                case UnicodeCategory.FinalQuotePunctuation:
                case UnicodeCategory.OtherPunctuation:
                case UnicodeCategory.MathSymbol:
                case UnicodeCategory.CurrencySymbol:
                case UnicodeCategory.ModifierSymbol:
                case UnicodeCategory.OtherSymbol:
                    continue;
                default:
                    return false;
            }
        }
        return true;
    }

    private static int CodePoints(string text)
    {
        int count = 0;
        for (int i = 0; i < text.Length; i += char.IsSurrogatePair(text, i) ? 2 : 1) count++;
        return count;
    }
}
