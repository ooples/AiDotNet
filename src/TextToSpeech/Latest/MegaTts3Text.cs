using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// MegaTTS 3's text chunking (<c>tts/utils/text_utils/split_text.py</c>) and the crossfade that joins the chunks' audio
/// (<c>tts/utils/audio_utils/io.py</c>), as the release computes them (Python's code points, UTF-8 lengths, <c>\s</c>
/// and <c>str.strip</c>).
/// </summary>
internal static class MegaTts3Text
{
    // Python's \s and str.isspace() (Unicode whitespace and the separators \x1c-\x1f).
    private const string PythonSpace = "[\t\n\v\f\r\u001c-\u001f \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]";
    private static readonly Regex SentenceBreak = new("(?<=[;:,.!?])" + PythonSpace + "+|(?<=[\uff1b\uff1a\uff0c\u3002\uff01\uff1f])", RegexOptions.CultureInvariant);

    /// <summary><c>chunk_text_english</c>: sentences (split after punctuation and whitespace, or after full-width
    /// punctuation) gathered into chunks of at most <paramref name="maxBytes"/> UTF-8 bytes, a space after each sentence
    /// that ends in a one-byte character, each chunk stripped.</summary>
    public static List<string> ChunkEnglish(string text, int maxBytes = 130)
    {
        var sentences = new List<string>();
        int start = 0;
        foreach (Match m in SentenceBreak.Matches(text))
        {
            sentences.Add(text.Substring(start, m.Index - start));
            start = m.Index + m.Length;
        }
        sentences.Add(text.Substring(start));
        string Joined(string sentence) => sentence.Length > 0 && Utf8Length(LastCodePoint(sentence)) == 1 ? sentence + " " : sentence;
        var chunks = new List<string>();
        string current = "";
        foreach (var sentence in sentences)
        {
            if (Utf8Length(current) + Utf8Length(sentence) <= maxBytes) current += Joined(sentence);
            else
            {
                if (current.Length > 0) chunks.Add(OuteTTSPrompts.PythonStrip(current));
                current = Joined(sentence);
            }
        }
        if (current.Length > 0) chunks.Add(OuteTTSPrompts.PythonStrip(current));
        return chunks;
    }

    private const string Primary = "\u3002.\uff01!\uff1f?";
    private const string Secondary = "\uff0c,\uff1b;\uff1a";
    private const string Tertiary = "\u3001\u2026\u2014-~\uff5e";

    /// <summary>
    /// <c>chunk_text_chinesev2</c>: when a chunk reaches <paramref name="limit"/> CJK ideographs it ends at the nearest
    /// sentence end ahead (within <paramref name="lookAhead"/> characters) or behind, else a comma, semicolon or colon, else
    /// an enumeration comma, ellipsis or dash, else where it is; then ASCII punctuation becomes full-width.
    /// </summary>
    public static List<string> ChunkChinese(string text, int limit = 60, int lookAhead = 30)
    {
        var chars = CodePoints(text);
        bool Ideograph(string c) => c.Length == 1 && c[0] >= '\u4e00' && c[0] <= '\u9fff';
        var result = new List<string>();
        var current = new List<string>();
        int count = 0;
        for (int i = 0; i < chars.Count; i++)
        {
            current.Add(chars[i]);
            if (Ideograph(chars[i])) count++;
            if (count < limit) continue;
            bool found = false;
            bool Ahead(string marks)
            {
                for (int k = 1; k < Math.Min(lookAhead, chars.Count - i); k++)
                {
                    if (!marks.Contains(chars[i + k])) continue;
                    result.Add(string.Concat(current) + string.Concat(chars.Skip(i + 1).Take(k)));
                    current = new List<string>();
                    count = 0;
                    i += k;
                    return true;
                }
                return false;
            }
            bool Behind(string marks)
            {
                for (int j = current.Count - 1; j >= 0; j--)
                {
                    if (!marks.Contains(current[j])) continue;
                    result.Add(string.Concat(current.Take(j + 1)));
                    current = current.Skip(j + 1).ToList();
                    count = current.Count(Ideograph);
                    return true;
                }
                return false;
            }
            found = Ahead(Primary) || Behind(Primary) || Ahead(Secondary) || Behind(Secondary) || Ahead(Tertiary) || Behind(Tertiary);
            if (!found)
            {
                result.Add(string.Concat(current));
                current = new List<string>();
                count = 0;
            }
        }
        if (current.Count > 0) result.Add(string.Concat(current));
        var map = new[] { (".", "\u3002"), (",", "\uff0c"), ("!", "\uff01"), ("?", "\uff1f"), (";", "\uff1b"), (":", "\uff1a") };
        for (int r = 0; r < result.Count; r++)
            foreach (var (ascii, full) in map) result[r] = result[r].Replace(ascii, full);
        return result;
    }

    /// <summary><c>combine_audio_segments</c>: each segment joined to the last over <paramref name="crossfade"/> seconds
    /// with the halves of a Hann (numpy <c>hanning</c>) window.</summary>
    public static double[] Combine(IReadOnlyList<double[]> segments, double crossfade = 0.16, int sampleRate = 24000)
    {
        if (segments.Count == 0) throw new ArgumentException("No segments to combine.", nameof(segments));
        int w = (int)(sampleRate * crossfade), n = 2 * w;
        var window = new double[n];
        for (int k = 0; k < n; k++) window[k] = n == 1 ? 1.0 : 0.5 - 0.5 * Math.Cos(2 * Math.PI * k / (n - 1));
        var combined = segments[0];
        for (int s = 1; s < segments.Count; s++)
        {
            var segment = segments[s];
            if (combined.Length < w || segment.Length < w) throw new ArgumentException("A segment is shorter than the crossfade.", nameof(segments));
            var next = new double[combined.Length + segment.Length - w];
            Array.Copy(combined, next, combined.Length - w);
            for (int k = 0; k < w; k++)
                next[combined.Length - w + k] = combined[combined.Length - w + k] * window[w + k] + segment[k] * window[k];
            Array.Copy(segment, w, next, combined.Length, segment.Length - w);
            combined = next;
        }
        return combined;
    }

    private static List<string> CodePoints(string text)
    {
        var points = new List<string>();
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])) { points.Add(text.Substring(i, 2)); i++; }
            else points.Add(text[i].ToString());
        }
        return points;
    }

    private static string LastCodePoint(string s) =>
        s.Length >= 2 && char.IsLowSurrogate(s[^1]) && char.IsHighSurrogate(s[^2]) ? s.Substring(s.Length - 2) : s.Substring(s.Length - 1);

    private static int Utf8Length(string s) => Encoding.UTF8.GetByteCount(s);
}
