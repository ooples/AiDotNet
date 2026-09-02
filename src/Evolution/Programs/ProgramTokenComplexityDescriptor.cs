using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Measures a candidate program by the number of lexical tokens in its normalized source.</summary>
/// <remarks>
/// <para>
/// Counting tokens rather than characters gives an archive axis that tracks structure instead of typography. A
/// model that renames <c>i</c> to <c>rowIndex</c>, wraps a long line, or adds indentation changes the character
/// count substantially and the token count not at all, so candidates stay in the cell their structure earns. The
/// tokenizer is language neutral and deterministic: it groups runs of letters, digits, underscores, and dots into
/// one identifier or number token, so <c>self.value</c> and <c>3.14</c> each count once; it treats a quoted run as
/// a single string token; and it counts every other non-white-space character as its own punctuation token.
/// </para>
/// <para>
/// Comments are counted like any other text because stripping them correctly would require a per-language parser,
/// and a silent, partially correct comment stripper would make the axis depend on which language guess was made.
/// The count is therefore an honest proxy for source size rather than an estimate of semantic complexity.
/// </para>
/// <para><b>For Beginners:</b> This descriptor asks "how many pieces is this program made of?" — where a piece is
/// a name, a number, a piece of text in quotes, or a symbol such as a bracket or plus sign. It is a better measure
/// of how big a program really is than counting characters, because renaming a variable to something longer does
/// not change it. Use it as one axis of a quality-diversity grid to keep both compact and elaborate solutions
/// alive in the archive.</para>
/// </remarks>
public sealed class ProgramTokenComplexityDescriptor : IProgramDescriptor
{
    /// <summary>The descriptor name used when none is supplied.</summary>
    public const string DefaultName = "tokenComplexity";

    /// <summary>Initializes a token-count descriptor.</summary>
    /// <param name="name">The archive dimension name this descriptor fills.</param>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty or white space.</exception>
    public ProgramTokenComplexityDescriptor(string name = DefaultName)
    {
        Guard.NotNullOrWhiteSpace(name);
        Name = name.Trim();
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public double Compute(ProgramGenome genome)
    {
        Guard.NotNull(genome);
        return CountTokens(genome.NormalizedSource);
    }

    /// <summary>Counts the lexical tokens in a source snippet using this descriptor's tokenizer.</summary>
    /// <param name="source">The text to tokenize.</param>
    /// <returns>The number of identifier, number, string, and punctuation tokens.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static int CountTokens(string source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));

        int tokens = 0;
        int index = 0;
        while (index < source.Length)
        {
            char current = source[index];
            if (char.IsWhiteSpace(current))
            {
                index++;
                continue;
            }

            if (current == '"' || current == '\'' || current == '`')
            {
                index = SkipQuoted(source, index, current);
                tokens++;
                continue;
            }

            if (char.IsLetterOrDigit(current) || current == '_')
            {
                while (index < source.Length
                    && (char.IsLetterOrDigit(source[index]) || source[index] == '_' || source[index] == '.'))
                {
                    index++;
                }

                tokens++;
                continue;
            }

            index++;
            tokens++;
        }

        return tokens;
    }

    private static int SkipQuoted(string source, int start, char quote)
    {
        int index = start + 1;
        while (index < source.Length)
        {
            char current = source[index];
            if (current == '\\')
            {
                index += 2;
                continue;
            }

            if (current == quote) return index + 1;
            if (current == '\n') return index;
            index++;
        }

        return index;
    }
}
