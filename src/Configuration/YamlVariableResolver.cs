using System.Text;

namespace AiDotNet.Configuration;

/// <summary>Substitutes <c>${NAME}</c> references in a YAML document before it is deserialized.</summary>
/// <remarks>
/// <para>
/// A configuration file is meant to be checked in, and some of what a run needs must not be: an API key, a licence
/// token, a machine-specific output directory, a run identifier that differs per environment. This resolver lets the
/// file name those values instead of containing them. <c>${NAME}</c> is replaced by the environment variable
/// <c>NAME</c>; <c>${NAME:-fallback}</c> uses <c>fallback</c> when the variable is unset or empty; and
/// <c>$${NAME}</c> is an escape that produces the literal text <c>${NAME}</c>. A reference with no value and no
/// fallback is an error naming the variable, because the alternative - quietly substituting an empty string - turns a
/// missing API key into an authentication failure hundreds of lines away from its cause.
/// </para>
/// <para>
/// Substitution happens on the document text, before parsing, so a reference can appear anywhere: in a value, in a
/// key, or inside a longer string. That also means a substituted value carrying YAML syntax can change the shape of
/// the document, so quote the value in the file when it might contain a colon, a hash, or a newline.
/// </para>
/// <para><b>For Beginners:</b> Write <c>apiKey: ${OPENAI_API_KEY}</c> in your YAML file and set the
/// <c>OPENAI_API_KEY</c> environment variable before running. The secret stays out of the file, and the file is safe
/// to commit. Use <c>${LOG_DIR:-./runs}</c> when you want a sensible default that a machine can override.</para>
/// </remarks>
public static class YamlVariableResolver
{
    /// <summary>Replaces every <c>${NAME}</c> reference using the process environment.</summary>
    /// <param name="content">The YAML document text.</param>
    /// <returns>The text with every reference resolved.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="content"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A reference has neither a value nor a fallback.</exception>
    public static string Resolve(string content) =>
        Resolve(content, name => Environment.GetEnvironmentVariable(name));

    /// <summary>Replaces every <c>${NAME}</c> reference using a caller-supplied lookup.</summary>
    /// <param name="content">The YAML document text.</param>
    /// <param name="lookup">Returns the value for a name, or <c>null</c> when it is not set.</param>
    /// <returns>The text with every reference resolved.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="content"/> or <paramref name="lookup"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A reference has neither a value nor a fallback.</exception>
    public static string Resolve(string content, Func<string, string?> lookup)
    {
        if (content is null) throw new ArgumentNullException(nameof(content));
        if (lookup is null) throw new ArgumentNullException(nameof(lookup));
        if (content.IndexOf("${", StringComparison.Ordinal) < 0) return content;

        var builder = new StringBuilder(content.Length);
        int index = 0;
        while (index < content.Length)
        {
            int start = content.IndexOf("${", index, StringComparison.Ordinal);
            if (start < 0)
            {
                builder.Append(content, index, content.Length - index);
                break;
            }

            // "$${NAME}" is the escape: emit the reference verbatim and consume the extra dollar sign.
            bool escaped = start > 0 && content[start - 1] == '$';
            int close = content.IndexOf('}', start + 2);
            if (close < 0)
            {
                builder.Append(content, index, content.Length - index);
                break;
            }

            string reference = content.Substring(start + 2, close - start - 2);
            if (!IsWellFormed(reference, out string name, out string? fallback))
            {
                builder.Append(content, index, close - index + 1);
                index = close + 1;
                continue;
            }

            if (escaped)
            {
                // The trailing dollar of the escape pair was already appended, so drop it and keep the literal.
                builder.Append(content, index, start - 1 - index);
                builder.Append(content, start, close - start + 1);
                index = close + 1;
                continue;
            }

            builder.Append(content, index, start - index);
            string? value = lookup(name);
            if (string.IsNullOrEmpty(value)) value = fallback;
            if (value is null)
                throw new ArgumentException(
                    $"The configuration references '${{{name}}}', but no environment variable named '{name}' is set " +
                    $"and the reference has no fallback. Set the variable, or write '${{{name}:-default}}'.",
                    nameof(content));

            builder.Append(value);
            index = close + 1;
        }

        return builder.ToString();
    }

    /// <summary>Splits a reference body into a name and an optional <c>:-</c> fallback.</summary>
    /// <param name="reference">The text between the braces.</param>
    /// <param name="name">The variable name when the reference is well formed.</param>
    /// <param name="fallback">The fallback when one was supplied, otherwise <c>null</c>.</param>
    /// <returns><c>true</c> when the reference names a valid identifier.</returns>
    /// <remarks>
    /// Anything that is not an identifier is left alone rather than rejected, so a document that happens to contain
    /// brace syntax of its own - a template, a regular expression, a shell snippet - passes through untouched.
    /// </remarks>
    private static bool IsWellFormed(string reference, out string name, out string? fallback)
    {
        name = reference;
        fallback = null;
        int separator = reference.IndexOf(":-", StringComparison.Ordinal);
        if (separator >= 0)
        {
            name = reference.Substring(0, separator);
            fallback = reference.Substring(separator + 2);
        }

        if (name.Length == 0) return false;
        if (!char.IsLetter(name[0]) && name[0] != '_') return false;
        foreach (char character in name)
            if (!char.IsLetterOrDigit(character) && character != '_') return false;
        return true;
    }
}
