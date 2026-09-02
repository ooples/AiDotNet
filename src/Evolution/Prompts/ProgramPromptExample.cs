using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Prompts;

/// <summary>One other program shown in a prompt, together with why it was chosen and how it scored.</summary>
/// <remarks>
/// <para>
/// A prompt shows other programs for two distinct reasons: to demonstrate what currently scores well, and to
/// demonstrate that other shapes of solution exist. <see cref="Kind"/> records which, so the model is told what
/// an example is for rather than left to infer it. The reference OpenEvolve implementation instead labels an
/// example from its score band whenever no metadata flag is set, which presents a deliberately diverse
/// low-scoring program to the model as "Exploratory" — a judgement about its quality rather than a statement
/// about its role.
/// </para>
/// <para>
/// <see cref="Descriptors"/> carries the archive coordinates that placed this program in its cell, which is what
/// lets the prompt say <em>how</em> an example differs rather than merely that it does.
/// </para>
/// <para><b>For Beginners:</b> This is one of the other programs the AI is shown alongside yours, bundled with
/// its score, its measured characteristics, and a note about why it is in the prompt — because it scored well,
/// because it solves the problem differently, or because it drifted in from elsewhere in the search. Giving the
/// AI a couple of good examples and a couple of different ones is what stops it from making the same small tweak
/// over and over.</para>
/// </remarks>
public sealed class ProgramPromptExample
{
    private static readonly Dictionary<string, double> NoDescriptors = new(StringComparer.Ordinal);

    /// <summary>Initializes an example program.</summary>
    /// <param name="genome">The program being shown.</param>
    /// <param name="kind">Why this example is in the prompt.</param>
    /// <param name="quality">Its fitness score, or <c>null</c> when it was never scored.</param>
    /// <param name="descriptors">Its archive coordinates, or <c>null</c> for none.</param>
    /// <param name="changesDescription">A short note about what it changed, or <c>null</c> for none.</param>
    /// <exception cref="ArgumentNullException"><paramref name="genome"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="kind"/> is not a defined value.</exception>
    /// <exception cref="ArgumentException"><paramref name="quality"/> is not a finite number.</exception>
    public ProgramPromptExample(
        ProgramGenome genome,
        ProgramPromptExampleKind kind = ProgramPromptExampleKind.TopProgram,
        double? quality = null,
        IReadOnlyDictionary<string, double>? descriptors = null,
        string? changesDescription = null)
    {
        Guard.NotNull(genome);
        if (!Enum.IsDefined(typeof(ProgramPromptExampleKind), kind))
        {
            throw new ArgumentOutOfRangeException(nameof(kind), kind, "Value must be a defined example kind.");
        }

        if (quality.HasValue && (double.IsNaN(quality.Value) || double.IsInfinity(quality.Value)))
        {
            throw new ArgumentException("An example's quality must be a finite number.", nameof(quality));
        }

        Genome = genome;
        Kind = kind;
        Quality = quality;
        Descriptors = descriptors is null
            ? NoDescriptors
            : new Dictionary<string, double>(CopyOf(descriptors), StringComparer.Ordinal);
        ChangesDescription = changesDescription;
    }

    /// <summary>Gets the program being shown.</summary>
    public ProgramGenome Genome { get; }

    /// <summary>Gets why this example is in the prompt.</summary>
    public ProgramPromptExampleKind Kind { get; }

    /// <summary>Gets the example's fitness score, or <c>null</c> when it was never scored.</summary>
    public double? Quality { get; }

    /// <summary>Gets the example's archive coordinates; empty when none were supplied.</summary>
    public IReadOnlyDictionary<string, double> Descriptors { get; }

    /// <summary>Gets a short note about what this example changed, or <c>null</c> for none.</summary>
    public string? ChangesDescription { get; }

    /// <summary>Returns a description that never echoes the program source.</summary>
    /// <returns>The kind, the genome identity, and the score.</returns>
    public override string ToString() =>
        $"ProgramPromptExample({Kind}, id={Genome.Id.Substring(0, Math.Min(12, Genome.Id.Length))}, quality={Quality})";

    private static Dictionary<string, double> CopyOf(IReadOnlyDictionary<string, double> source)
    {
        var copy = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> pair in source)
        {
            if (pair.Key is null) throw new ArgumentException("A descriptor name cannot be null.", nameof(source));
            copy[pair.Key] = pair.Value;
        }

        return copy;
    }
}
