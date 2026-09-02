using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Measures a candidate program by the character length of its normalized source.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation uses <c>len(program.code)</c> as its built-in complexity axis, which
/// makes the coordinate move when a model reformats a file: adding CRLF line endings inflates it by one character
/// per line, and trailing spaces inflate it further, so the same program can occupy two different archive cells.
/// This descriptor measures <see cref="ProgramGenome.NormalizedSource"/> instead, which has already had its line
/// endings unified and its trailing white space removed, so cosmetic edits cannot move a candidate between cells.
/// </para>
/// <para><b>For Beginners:</b> Quality-diversity search sorts candidates into pigeonholes so it keeps a variety of
/// good answers. This descriptor supplies one common pigeonhole axis: how long the program is. Short and long
/// solutions then compete only against others of similar size, which stops one very long solution from crowding
/// out an elegant short one. Pair it with an
/// <see cref="EvolutionDescriptorDefinition"/> whose bounds cover the program sizes you expect.</para>
/// </remarks>
public sealed class ProgramLengthDescriptor : IProgramDescriptor
{
    /// <summary>The descriptor name used when none is supplied.</summary>
    public const string DefaultName = "length";

    /// <summary>Initializes a length descriptor.</summary>
    /// <param name="name">The archive dimension name this descriptor fills.</param>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty or white space.</exception>
    public ProgramLengthDescriptor(string name = DefaultName)
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
        return genome.NormalizedSource.Length;
    }
}
