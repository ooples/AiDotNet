namespace AiDotNet.Configuration;

/// <summary>
/// Configures quality-diversity search when AutoML uses the MAP-Elites strategy.
/// </summary>
/// <remarks>
/// <para>
/// MAP-Elites retains the best validated model specification in each behavior cell instead of
/// retaining only one globally best configuration. The archive stores specifications and scores,
/// not trained model instances, so archive growth does not retain one live model per cell.
/// </para>
/// <para><b>For Beginners:</b> These settings control how broadly AutoML explores different model
/// families and configuration-complexity levels while still returning the best model normally.</para>
/// </remarks>
public sealed class MapElitesAutoMLOptions
{
    /// <summary>Gets or sets the stable root seed used to derive proposal-local random streams.</summary>
    public ulong Seed { get; set; } = 1234UL;

    /// <summary>Gets or sets the minimum number of random seed configurations.</summary>
    /// <remarks>
    /// When the trial budget permits, the implementation raises this value internally to cover every
    /// configured model family at least once.
    /// </remarks>
    public int InitialPopulationSize { get; set; } = 8;

    /// <summary>Gets or sets the number of bins for normalized configuration complexity.</summary>
    public int ComplexityBinCount { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum occupied archive cells, or zero to use the complete descriptor grid.
    /// </summary>
    public int ArchiveCapacity { get; set; } = 128;

    /// <summary>Gets or sets the probability that a mutable parameter is resampled.</summary>
    public double MutationProbability { get; set; } = 0.25;

    /// <summary>Gets or sets the probability of proposing a fresh model family and configuration.</summary>
    public double ExplorationProbability { get; set; } = 0.15;

    /// <summary>Gets or sets how many elite configurations may inform one proposal.</summary>
    public int InspirationCount { get; set; } = 3;

    /// <summary>
    /// Gets or sets the proposal allowance per expensive trial attempt.
    /// </summary>
    /// <remarks>
    /// Duplicate configurations are rejected before training and therefore do not consume
    /// <c>TrialLimit</c>. This multiplier bounds how long the engine may search for another unique
    /// configuration after the search space becomes saturated.
    /// </remarks>
    public int MaxProposalMultiplier { get; set; } = 20;

    internal MapElitesAutoMLOptions SnapshotAndValidate()
    {
        if (InitialPopulationSize <= 0) throw new ArgumentOutOfRangeException(nameof(InitialPopulationSize));
        if (ComplexityBinCount <= 0) throw new ArgumentOutOfRangeException(nameof(ComplexityBinCount));
        if (ArchiveCapacity < 0) throw new ArgumentOutOfRangeException(nameof(ArchiveCapacity));
        ValidateProbability(MutationProbability, nameof(MutationProbability));
        ValidateProbability(ExplorationProbability, nameof(ExplorationProbability));
        if (InspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(InspirationCount));
        if (MaxProposalMultiplier <= 0) throw new ArgumentOutOfRangeException(nameof(MaxProposalMultiplier));

        return new MapElitesAutoMLOptions
        {
            Seed = Seed,
            InitialPopulationSize = InitialPopulationSize,
            ComplexityBinCount = ComplexityBinCount,
            ArchiveCapacity = ArchiveCapacity,
            MutationProbability = MutationProbability,
            ExplorationProbability = ExplorationProbability,
            InspirationCount = InspirationCount,
            MaxProposalMultiplier = MaxProposalMultiplier
        };
    }

    private static void ValidateProbability(double value, string parameterName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0 || value > 1)
            throw new ArgumentOutOfRangeException(parameterName, "Probabilities must be finite values in [0, 1].");
    }
}
