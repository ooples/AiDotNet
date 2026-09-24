using System.Globalization;
using System.Text.RegularExpressions;

namespace AiDotNet.Serving.Security;

/// <summary>
/// Neutralizes caller-controlled strings before they are written to a log entry (CWE-117, log forging).
/// </summary>
/// <remarks>
/// <para>
/// Microsoft.Extensions.Logging renders structured-logging arguments verbatim into the formatted
/// message, and plain-text sinks (the simple console formatter, file sinks, syslog forwarders) do not
/// escape them. A route value such as <c>"m\r\n[warn] admin login ok"</c> would therefore start a
/// forged log line. Every caller-controlled string passed to a logger in AiDotNet.Serving goes through
/// <see cref="Sanitize"/> so the value is still recorded, but can never break out of its own entry.
/// </para>
/// <para>
/// Escaped characters: all C0/C1 control characters (which includes CR, LF, TAB, NUL, ESC and NEL),
/// the Unicode line/paragraph separators U+2028/U+2029, and the bidirectional embedding/override/isolate
/// controls U+202A-U+202E and U+2066-U+2069 (which can visually reorder a log line). CR, LF and TAB are
/// rendered as <c>\r</c>, <c>\n</c> and <c>\t</c>; everything else as <c>\uXXXX</c>. Printable text,
/// including non-ASCII letters, is left untouched, so ordinary model names and paths log unchanged.
/// </para>
/// <para>
/// This is for log output only. Never feed the sanitized value back into lookups, file paths or
/// responses: the raw value remains the source of truth for request handling.
/// </para>
/// </remarks>
internal static class LogSanitizer
{
    private static readonly Regex UnsafeCharacters = new(
        @"[\p{Cc}\u2028\u2029\u202A-\u202E\u2066-\u2069]",
        RegexOptions.Compiled | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    /// <summary>
    /// Returns <paramref name="value"/> with every line-breaking or otherwise unsafe character escaped,
    /// or <see langword="null"/> when <paramref name="value"/> is <see langword="null"/>.
    /// </summary>
    /// <param name="value">A caller-controlled value that is about to be logged.</param>
    /// <returns>The value, safe to embed in a single log entry.</returns>
    public static string? Sanitize(string? value) =>
        value is null ? null : UnsafeCharacters.Replace(value, static match => Escape(match.Value[0]));

    private static string Escape(char c) => c switch
    {
        '\r' => "\\r",
        '\n' => "\\n",
        '\t' => "\\t",
        _ => "\\u" + ((int)c).ToString("X4", CultureInfo.InvariantCulture)
    };
}
