using System.Text.RegularExpressions;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Removes terminal control sequences and credential-shaped substrings from untrusted evaluator text.</summary>
/// <remarks>
/// <para>
/// Artifact and diagnostic text produced by an evaluated candidate is untrusted: it can echo environment variables,
/// connection strings, or request headers into a checkpoint, a log, or a prompt. <see cref="Sanitize"/> strips ANSI
/// escape sequences and then applies an ordered pattern table, <b>specific rules first</b>, so a provider-shaped key is
/// labelled as such instead of being swallowed by the generic long-token rule. That ordering is the difference from
/// OpenEvolve's filter (prompt/sampler.py:711-740), which places its generic <c>[A-Za-z0-9]{32,}</c> rule first and
/// therefore makes its own <c>sk-</c> rule unreachable and rewrites every content hash it sees.
/// </para>
/// <para>
/// The generic rule allows through anything shaped like a content digest - a run of exactly 40, 56, 64, 96, or 128
/// hexadecimal characters - because those are the canonical genome, task, and configuration identities this engine puts
/// in its own diagnostics, and redacting them would destroy the ability to correlate a failure with the candidate that
/// caused it. Every pattern is linear in the input length and carries a match timeout, and the input is already bounded
/// by <see cref="EvolutionArtifact.MaximumTextLength"/>, so sanitizing is O(n) with no backtracking blow-up.
/// </para>
/// <para><b>For Beginners:</b> Programs under evolution frequently print more than you expect - a stack trace can
/// include the command line, and a command line can include a password. Before any of that text is saved to disk or
/// shown to a language model, this helper rewrites the risky parts, so <c>token=abc123</c> becomes
/// <c>token=&lt;REDACTED&gt;</c>. It also removes the invisible colour codes terminals emit, which otherwise turn a log
/// into unreadable noise. It deliberately leaves long hexadecimal fingerprints alone, because those are identifiers you
/// need, not secrets. Sanitizing is on by default; you can turn it off only if you fully control the evaluated code.</para>
/// </remarks>
public static class EvolutionArtifactSanitizer
{
    private const string TokenReplacement = "<REDACTED_TOKEN>";

    /// <summary>The fail-safe replacement used when a pattern somehow exceeds <see cref="MatchTimeout"/>.</summary>
    private const string TimeoutReplacement = "<REDACTED_UNSCANNABLE_ARTIFACT>";

    private static readonly TimeSpan MatchTimeout = TimeSpan.FromSeconds(2);

    private static readonly Regex AnsiEscape = new(
        @"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", RegexOptions.CultureInvariant, MatchTimeout);

    private static readonly Regex ProviderApiKey = new(
        @"\bsk-[A-Za-z0-9_\-]{16,}", RegexOptions.CultureInvariant, MatchTimeout);

    private static readonly Regex AwsAccessKey = new(
        @"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|ABIA|ACCA)[0-9A-Z]{16}\b", RegexOptions.CultureInvariant, MatchTimeout);

    private static readonly Regex JsonWebToken = new(
        @"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", RegexOptions.CultureInvariant, MatchTimeout);

    private static readonly Regex BearerCredential = new(
        @"\b[Bb]earer\s+[A-Za-z0-9._\-+/=]{8,}", RegexOptions.CultureInvariant, MatchTimeout);

    private static readonly Regex NamedCredential = new(
        @"\b(password|passwd|pwd|secret|token|api[_\-]?key|access[_\-]?key|private[_\-]?key)\b\s*[=:]\s*[^\s""',;]+",
        RegexOptions.CultureInvariant | RegexOptions.IgnoreCase, MatchTimeout);

    private static readonly Regex LongToken = new(
        @"\b[A-Za-z0-9]{40,}\b", RegexOptions.CultureInvariant, MatchTimeout);

    private static readonly Regex HexadecimalDigest = new(
        @"^(?:[0-9a-f]+|[0-9A-F]+)$", RegexOptions.CultureInvariant, MatchTimeout);

    /// <summary>Returns <paramref name="text"/> with control sequences stripped and credential-shaped runs redacted.</summary>
    /// <param name="text">The untrusted text to sanitize.</param>
    /// <returns>
    /// The sanitized text. The result is reference-equal to the input only by coincidence; compare with
    /// <see cref="StringComparer.Ordinal"/> to learn whether anything was removed.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    public static string Sanitize(string text)
    {
        Guard.NotNull(text);
        if (text.Length == 0) return text;
        try
        {
            string filtered = AnsiEscape.Replace(text, string.Empty);
            filtered = ProviderApiKey.Replace(filtered, "<REDACTED_API_KEY>");
            filtered = AwsAccessKey.Replace(filtered, "<REDACTED_ACCESS_KEY>");
            filtered = JsonWebToken.Replace(filtered, "<REDACTED_JWT>");
            filtered = BearerCredential.Replace(filtered, "Bearer <REDACTED>");
            filtered = NamedCredential.Replace(filtered, match => match.Groups[1].Value + "=<REDACTED>");
            return LongToken.Replace(filtered, ReplaceUnlessDigest);
        }
        catch (RegexMatchTimeoutException)
        {
            // Every pattern is linear in the input and the input is length-bounded, so this is unreachable in
            // practice; if it ever happens, discarding the text is the safe direction for a secret scrubber.
            return TimeoutReplacement;
        }
    }

    /// <summary>Returns whether sanitizing <paramref name="text"/> would change it.</summary>
    /// <param name="text">The untrusted text to inspect.</param>
    /// <returns><c>true</c> when <see cref="Sanitize"/> would remove or rewrite something.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    public static bool WouldRedact(string text) => !string.Equals(Sanitize(text), text, StringComparison.Ordinal);

    private static string ReplaceUnlessDigest(Match match)
    {
        string value = match.Value;
        bool digestLength = value.Length is 40 or 56 or 64 or 96 or 128;
        return digestLength && HexadecimalDigest.IsMatch(value) ? value : TokenReplacement;
    }
}
