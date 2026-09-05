using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Prompts;

/// <summary>Strips terminal control sequences and credential-shaped text before it reaches a prompt or a log.</summary>
/// <remarks>
/// <para>
/// Everything a sandboxed program writes is untrusted: it is the output of code a model wrote, run against data
/// the library did not choose. Quoting that output back into the next prompt is what makes a search converge, but
/// it is also the path by which an API key printed into a stack trace, or an escape sequence that rewrites the
/// operator's terminal, would travel into a request body and a log file. This type is the single choke point that
/// text passes through on its way into a prompt.
/// </para>
/// <para>
/// Redaction is conservative by design: it removes ANSI escape sequences and every other control character except
/// tab and newline, and it replaces values shaped like bearer tokens, provider keys, and assignments to
/// password-like or token-like names. It cannot recognise a secret that looks like ordinary prose, so it is a
/// safety net rather than a guarantee — the primary defence remains not putting credentials where a sandboxed
/// program can read them.
/// </para>
/// <para><b>For Beginners:</b> Programs print all sorts of things when they run, and some of it should not be
/// forwarded to an AI provider or written into a log: passwords, API keys, and invisible characters that can mess
/// up a terminal. This helper cleans that text first, replacing anything that looks like a secret with
/// <c>&lt;redacted&gt;</c>. You normally never call it yourself — the prompt builder runs everything through it
/// automatically — but it is available if you assemble prompts of your own.</para>
/// </remarks>
public static class PromptTextRedactor
{
    /// <summary>The text substituted for a value that looked like a credential.</summary>
    public const string RedactionMarker = "<redacted>";

    private const int RegexTimeoutMilliseconds = 2_000;

    private static readonly RegexOptions Options =
        RegexOptions.CultureInvariant | RegexOptions.IgnoreCase;

    // "ESC [ ... final-byte" and the shorter "ESC single-char" forms. Terminal
    // escapes are removed rather than redacted: they carry no information a model
    // needs and can rewrite the operator's console when a log is tailed.
    private static readonly Regex AnsiEscape = new(
        @"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", Options, TimeSpan.FromMilliseconds(RegexTimeoutMilliseconds));

    private static readonly Regex ProviderKey = new(
        @"\b(?:sk|pk|rk|api|key)-[A-Za-z0-9_\-]{16,}\b", Options, TimeSpan.FromMilliseconds(RegexTimeoutMilliseconds));

    private static readonly Regex BearerToken = new(
        @"\bBearer\s+[A-Za-z0-9\-._~+/]{16,}=*", Options, TimeSpan.FromMilliseconds(RegexTimeoutMilliseconds));

    private static readonly Regex JsonWebToken = new(
        @"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b", Options, TimeSpan.FromMilliseconds(RegexTimeoutMilliseconds));

    private static readonly Regex SecretAssignment = new(
        @"\b(password|passwd|secret|token|api[_\-]?key|access[_\-]?key|client[_\-]?secret|authorization)\b\s*[=:]\s*\S+",
        Options, TimeSpan.FromMilliseconds(RegexTimeoutMilliseconds));

    // Long unbroken alphanumeric runs are the classic shape of an opaque token.
    // The bound is high enough that ordinary identifiers, hashes shown on purpose,
    // and base64 chunks of real output are left alone.
    private static readonly Regex OpaqueToken = new(
        @"(?<![A-Za-z0-9])[A-Za-z0-9]{40,}(?![A-Za-z0-9])", Options, TimeSpan.FromMilliseconds(RegexTimeoutMilliseconds));

    /// <summary>Removes control sequences and credential-shaped values from text.</summary>
    /// <param name="text">The untrusted text.</param>
    /// <returns>The cleaned text; never <c>null</c>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    public static string Redact(string text)
    {
        Guard.NotNull(text);
        if (text.Length == 0) return string.Empty;

        string result = StripControlCharacters(text);
        try
        {
            result = SecretAssignment.Replace(result, match => match.Groups[1].Value + "=" + RedactionMarker);
            result = BearerToken.Replace(result, "Bearer " + RedactionMarker);
            result = JsonWebToken.Replace(result, RedactionMarker);
            result = ProviderKey.Replace(result, RedactionMarker);
            result = OpaqueToken.Replace(result, RedactionMarker);
        }
        catch (RegexMatchTimeoutException)
        {
            // Adversarial output must never stall the run. When a pattern cannot
            // finish in time the text is not proven safe, so none of it is quoted.
            return RedactionMarker;
        }

        return result;
    }

    /// <summary>Redacts text and truncates it to a byte budget, appending a marker when it was cut.</summary>
    /// <param name="text">The untrusted text.</param>
    /// <param name="maxBytes">The maximum number of UTF-8 bytes the result may occupy before the marker.</param>
    /// <param name="truncationMarker">The text appended when truncation occurred; may be empty.</param>
    /// <param name="wasTruncated">Set to <c>true</c> when the text did not fit.</param>
    /// <returns>The cleaned, bounded text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> or <paramref name="truncationMarker"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxBytes"/> is negative.</exception>
    public static string RedactAndBound(string text, int maxBytes, string truncationMarker, out bool wasTruncated)
    {
        Guard.NotNull(text);
        Guard.NotNull(truncationMarker);
        if (maxBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBytes), maxBytes, "Value cannot be negative.");
        }

        string redacted = Redact(text);
        string bounded = BoundToUtf8Bytes(redacted, maxBytes, out wasTruncated);
        return wasTruncated && truncationMarker.Length > 0
            ? bounded + (bounded.Length > 0 ? "\n" : string.Empty) + truncationMarker
            : bounded;
    }

    /// <summary>Truncates text so its UTF-8 encoding fits a byte budget without splitting a character.</summary>
    /// <param name="text">The text to bound.</param>
    /// <param name="maxBytes">The maximum number of UTF-8 bytes.</param>
    /// <param name="wasTruncated">Set to <c>true</c> when characters were dropped.</param>
    /// <returns>The bounded text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxBytes"/> is negative.</exception>
    public static string BoundToUtf8Bytes(string text, int maxBytes, out bool wasTruncated)
    {
        Guard.NotNull(text);
        if (maxBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBytes), maxBytes, "Value cannot be negative.");
        }

        wasTruncated = false;
        if (maxBytes == 0)
        {
            wasTruncated = text.Length > 0;
            return string.Empty;
        }

        var encoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        if (encoding.GetByteCount(text) <= maxBytes) return text;

        wasTruncated = true;
        int used = 0;
        int index = 0;
        while (index < text.Length)
        {
            // Keep surrogate pairs together: cutting between the halves would emit
            // a lone surrogate that no UTF-8 encoder can represent.
            int width = char.IsHighSurrogate(text[index]) && index + 1 < text.Length && char.IsLowSurrogate(text[index + 1])
                ? 2
                : 1;
            int cost = encoding.GetByteCount(text.Substring(index, width));
            if (used + cost > maxBytes) break;
            used += cost;
            index += width;
        }

        return text.Substring(0, index);
    }

    /// <summary>Formats a byte count for a truncation notice using the invariant culture.</summary>
    /// <param name="bytes">The byte count.</param>
    /// <returns>The formatted count.</returns>
    public static string FormatBytes(int bytes) => bytes.ToString(CultureInfo.InvariantCulture);

    private static string StripControlCharacters(string text)
    {
        string withoutEscapes;
        try
        {
            withoutEscapes = AnsiEscape.Replace(text, string.Empty);
        }
        catch (RegexMatchTimeoutException)
        {
            return RedactionMarker;
        }

        var builder = new StringBuilder(withoutEscapes.Length);
        foreach (char character in withoutEscapes)
        {
            if (character == '\n' || character == '\t')
            {
                builder.Append(character);
                continue;
            }

            if (character == '\r') continue;
            if (char.IsControl(character)) continue;
            builder.Append(character);
        }

        return builder.ToString();
    }
}
