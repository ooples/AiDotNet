using System.Text;

namespace AiDotNet.Evolution.Programs;

/// <summary>Line-ending aware text helpers shared by the program-evolution substrate.</summary>
/// <remarks>
/// Every helper is deterministic, culture independent, and allocation conscious, because these routines run on
/// every parsed model response and on every genome identity computation.
/// </remarks>
internal static class ProgramText
{
    internal const char LineFeed = '\n';
    internal const string LineFeedText = "\n";
    internal const string CarriageReturnLineFeed = "\r\n";

    /// <summary>Splits text into lines on CRLF, CR, or LF without keeping the terminators.</summary>
    internal static List<string> SplitLines(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var lines = new List<string>();
        int start = 0;
        for (int index = 0; index < text.Length; index++)
        {
            char current = text[index];
            if (current == '\r')
            {
                lines.Add(text.Substring(start, index - start));
                if (index + 1 < text.Length && text[index + 1] == LineFeed) index++;
                start = index + 1;
            }
            else if (current == LineFeed)
            {
                lines.Add(text.Substring(start, index - start));
                start = index + 1;
            }
        }

        lines.Add(text.Substring(start, text.Length - start));
        return lines;
    }

    /// <summary>Returns the dominant line terminator of <paramref name="text"/>, defaulting to a line feed.</summary>
    internal static string DetectNewLine(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return text.IndexOf(CarriageReturnLineFeed, StringComparison.Ordinal) >= 0
            ? CarriageReturnLineFeed
            : LineFeedText;
    }

    /// <summary>Joins lines with <paramref name="newLine"/>, appending a terminator when requested.</summary>
    internal static string JoinLines(IReadOnlyList<string> lines, string newLine, bool trailingNewLine)
    {
        if (lines is null) throw new ArgumentNullException(nameof(lines));
        if (newLine is null) throw new ArgumentNullException(nameof(newLine));
        var builder = new StringBuilder();
        for (int index = 0; index < lines.Count; index++)
        {
            if (index > 0) builder.Append(newLine);
            builder.Append(lines[index]);
        }

        if (trailingNewLine) builder.Append(newLine);
        return builder.ToString();
    }

    /// <summary>Reports whether <paramref name="text"/> ends with a line terminator.</summary>
    internal static bool EndsWithNewLine(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (text.Length == 0) return false;
        char last = text[text.Length - 1];
        return last == LineFeed || last == '\r';
    }

    /// <summary>Removes a leading byte-order mark, normalizes terminators to line feeds, and trims line ends.</summary>
    internal static string Normalize(string source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        string text = source.Length > 0 && source[0] == '\uFEFF' ? source.Substring(1) : source;
        List<string> lines = SplitLines(text);
        for (int index = 0; index < lines.Count; index++) lines[index] = lines[index].TrimEnd();

        int lastContent = lines.Count - 1;
        while (lastContent >= 0 && lines[lastContent].Length == 0) lastContent--;
        if (lastContent < lines.Count - 1) lines.RemoveRange(lastContent + 1, lines.Count - lastContent - 1);
        return JoinLines(lines, LineFeedText, trailingNewLine: false);
    }

    /// <summary>Collapses every run of white space to one space and trims the result.</summary>
    internal static string CollapseWhitespace(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var builder = new StringBuilder(text.Length);
        bool pendingSpace = false;
        foreach (char character in text)
        {
            if (char.IsWhiteSpace(character))
            {
                pendingSpace = builder.Length > 0;
                continue;
            }

            if (pendingSpace)
            {
                builder.Append(' ');
                pendingSpace = false;
            }

            builder.Append(character);
        }

        return builder.ToString();
    }

    /// <summary>Truncates <paramref name="text"/> to <paramref name="maximumLength"/> characters with an ellipsis.</summary>
    internal static string Bound(string text, int maximumLength)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (maximumLength <= 0) return string.Empty;
        if (text.Length <= maximumLength) return text;
        return maximumLength <= 3
            ? text.Substring(0, maximumLength)
            : text.Substring(0, maximumLength - 3) + "...";
    }

    /// <summary>Replaces control characters other than tab with a middle dot so logs stay printable.</summary>
    internal static string Sanitize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var builder = new StringBuilder(text.Length);
        foreach (char character in text)
        {
            if (character == '\t' || character == LineFeed) builder.Append(' ');
            else if (char.IsControl(character)) builder.Append('\u00B7');
            else builder.Append(character);
        }

        return builder.ToString();
    }
}
