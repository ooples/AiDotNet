using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Prompts;

/// <summary>Everything one prompt is built from: the parent program, its scores, its history, and its evidence.</summary>
/// <remarks>
/// <para>
/// This is a plain input record with no engine types in it, which is deliberate. Because the prompt builder reads
/// only this object, prompt construction is a pure function that can be unit tested with a handful of literals —
/// no archive, no engine, no chat client, no clock. It is also what makes a rendered prompt reproducible: the
/// same context, options, and random stream produce the same characters every time, which is the property a
/// benchmark needs and the reference OpenEvolve sampler does not have, since it reads process-global state for
/// its diverse-program sampling and its template variations.
/// </para>
/// <para>
/// <see cref="FeatureDimensions"/>, <see cref="FeatureBins"/>, and <see cref="FeatureBinCounts"/> travel together
/// and are rendered as <c>name=value [bin 3/10]</c>, so the model can see where in the archive grid the parent
/// sits and which direction has room. Upstream renders the value alone, which tells the model a number but not
/// what the number means.
/// </para>
/// <para><b>For Beginners:</b> This is the packet of information handed to the prompt builder: the program being
/// improved, how well it scored, what it measured, the last few things that were tried, a few other programs
/// worth looking at, and anything the program printed when it last ran. Fill in what you have and leave the rest
/// alone — every part is optional except the program itself, and sections with nothing in them are simply left
/// out of the prompt.</para>
/// </remarks>
public sealed class ProgramPromptContext
{
    private static readonly Dictionary<string, double> NoDescriptors = new(StringComparer.Ordinal);

    /// <summary>Initializes a prompt context around the program being improved.</summary>
    /// <param name="parent">The program the model is asked to improve.</param>
    /// <exception cref="ArgumentNullException"><paramref name="parent"/> is <c>null</c>.</exception>
    public ProgramPromptContext(ProgramGenome parent)
    {
        Guard.NotNull(parent);
        Parent = parent;
    }

    /// <summary>Gets the program the model is asked to improve.</summary>
    public ProgramGenome Parent { get; }

    /// <summary>Gets or sets the parent's fitness score, or <c>null</c> when it was never scored.</summary>
    public double? ParentQuality { get; set; }

    /// <summary>Gets or sets whether a higher or a lower score is better.</summary>
    public EvolutionOptimizationDirection Direction { get; set; } = EvolutionOptimizationDirection.Maximize;

    /// <summary>Gets or sets the parent's measured values, keyed by name.</summary>
    public IReadOnlyDictionary<string, double> ParentMetrics { get; set; } = NoDescriptors;

    /// <summary>Gets or sets the parent's archive coordinates, keyed by descriptor name.</summary>
    public IReadOnlyDictionary<string, double> ParentDescriptors { get; set; } = NoDescriptors;

    /// <summary>Gets or sets the descriptor names the archive keeps diversity across, in grid order.</summary>
    public IReadOnlyList<string> FeatureDimensions { get; set; } = new List<string>();

    /// <summary>Gets or sets the parent's bin index along each dimension, aligned with <see cref="FeatureDimensions"/>.</summary>
    public IReadOnlyList<int> FeatureBins { get; set; } = new List<int>();

    /// <summary>Gets or sets the number of bins along each dimension, aligned with <see cref="FeatureDimensions"/>.</summary>
    public IReadOnlyList<int> FeatureBinCounts { get; set; } = new List<int>();

    /// <summary>Gets or sets the score of the immediately preceding attempt, used to say whether fitness moved.</summary>
    public double? PreviousQuality { get; set; }

    /// <summary>Gets or sets the earlier attempts summarized in the history section, oldest first.</summary>
    public IReadOnlyList<ProgramPromptAttempt> PreviousAttempts { get; set; } = new List<ProgramPromptAttempt>();

    /// <summary>Gets or sets the high-scoring programs offered as examples, best first.</summary>
    public IReadOnlyList<ProgramPromptExample> TopPrograms { get; set; } = new List<ProgramPromptExample>();

    /// <summary>Gets or sets the programs offered for their difference rather than their score.</summary>
    public IReadOnlyList<ProgramPromptExample> Inspirations { get; set; } = new List<ProgramPromptExample>();

    /// <summary>Gets or sets the untrusted output the parent's evaluation produced.</summary>
    public IReadOnlyList<ProgramPromptArtifact> Artifacts { get; set; } = new List<ProgramPromptArtifact>();

    /// <summary>Gets or sets the diagnostics the parent's evaluation reported.</summary>
    public IReadOnlyList<EvolutionDiagnostic> Diagnostics { get; set; } = new List<EvolutionDiagnostic>();

    /// <summary>Gets or sets descriptions of nearby unoccupied archive cells worth reaching.</summary>
    /// <remarks>Rendered only when coverage hints are enabled; upstream has no equivalent guidance.</remarks>
    public IReadOnlyList<string> EmptyNeighborCells { get; set; } = new List<string>();

    /// <summary>Gets or sets the parent's current change description, used in changes-description mode.</summary>
    public string? ChangesDescription { get; set; }

    /// <summary>Validates that the aligned feature lists agree and that no collection holds a null element.</summary>
    /// <exception cref="ArgumentException">
    /// A collection is <c>null</c> or holds a <c>null</c> element, a score is not finite, or
    /// <see cref="FeatureBins"/> or <see cref="FeatureBinCounts"/> is non-empty and does not match the length of
    /// <see cref="FeatureDimensions"/>.
    /// </exception>
    public void Validate()
    {
        RequireFinite(ParentQuality, nameof(ParentQuality));
        RequireFinite(PreviousQuality, nameof(PreviousQuality));

        if (ParentMetrics is null) throw new ArgumentException("ParentMetrics cannot be null.", nameof(ParentMetrics));
        if (ParentDescriptors is null) throw new ArgumentException("ParentDescriptors cannot be null.", nameof(ParentDescriptors));
        if (FeatureDimensions is null) throw new ArgumentException("FeatureDimensions cannot be null.", nameof(FeatureDimensions));
        if (FeatureBins is null) throw new ArgumentException("FeatureBins cannot be null.", nameof(FeatureBins));
        if (FeatureBinCounts is null) throw new ArgumentException("FeatureBinCounts cannot be null.", nameof(FeatureBinCounts));
        if (PreviousAttempts is null) throw new ArgumentException("PreviousAttempts cannot be null.", nameof(PreviousAttempts));
        if (TopPrograms is null) throw new ArgumentException("TopPrograms cannot be null.", nameof(TopPrograms));
        if (Inspirations is null) throw new ArgumentException("Inspirations cannot be null.", nameof(Inspirations));
        if (Artifacts is null) throw new ArgumentException("Artifacts cannot be null.", nameof(Artifacts));
        if (Diagnostics is null) throw new ArgumentException("Diagnostics cannot be null.", nameof(Diagnostics));
        if (EmptyNeighborCells is null) throw new ArgumentException("EmptyNeighborCells cannot be null.", nameof(EmptyNeighborCells));

        foreach (string dimension in FeatureDimensions)
        {
            if (dimension is null) throw new ArgumentException("A feature dimension name cannot be null.", nameof(FeatureDimensions));
        }

        if (FeatureBins.Count > 0 && FeatureBins.Count != FeatureDimensions.Count)
        {
            throw new ArgumentException(
                "FeatureBins must be empty or have one entry per feature dimension.", nameof(FeatureBins));
        }

        if (FeatureBinCounts.Count > 0 && FeatureBinCounts.Count != FeatureDimensions.Count)
        {
            throw new ArgumentException(
                "FeatureBinCounts must be empty or have one entry per feature dimension.", nameof(FeatureBinCounts));
        }

        RequireNoNulls(PreviousAttempts, nameof(PreviousAttempts));
        RequireNoNulls(TopPrograms, nameof(TopPrograms));
        RequireNoNulls(Inspirations, nameof(Inspirations));
        RequireNoNulls(Artifacts, nameof(Artifacts));
        RequireNoNulls(Diagnostics, nameof(Diagnostics));

        foreach (string cell in EmptyNeighborCells)
        {
            if (cell is null) throw new ArgumentException("An empty-cell description cannot be null.", nameof(EmptyNeighborCells));
        }
    }

    private static void RequireNoNulls<TItem>(IReadOnlyList<TItem> items, string parameterName)
        where TItem : class
    {
        for (int index = 0; index < items.Count; index++)
        {
            if (items[index] is null)
            {
                throw new ArgumentException($"{parameterName} cannot contain a null element.", parameterName);
            }
        }
    }

    private static void RequireFinite(double? value, string parameterName)
    {
        if (!value.HasValue) return;
        if (double.IsNaN(value.Value) || double.IsInfinity(value.Value))
        {
            throw new ArgumentException($"{parameterName} must be a finite number.", parameterName);
        }
    }
}
