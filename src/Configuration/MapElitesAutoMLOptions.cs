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
/// <para><b>For Beginners:</b> Ordinary AutoML keeps chasing the single best score, which tends to funnel
/// every trial into one model family and one region of settings. MAP-Elites instead lays out a grid whose
/// axes are the model family and how complex a configuration is, and it keeps the best-scoring specification
/// found in each grid cell, so at the end you can see the strongest simple linear model, the strongest
/// mid-complexity tree ensemble, and so on, while the overall winner is still returned normally. These
/// settings control that exploration: how many random seed configurations to try first, how finely to slice
/// complexity, how often to mutate or explore instead of refining known elites, and how many elites may
/// inspire one new proposal. Attach an instance through
/// <see cref="AutoMLOptions{T,TInput,TOutput}.MapElites"/> when <c>SearchStrategy</c> is
/// <c>AutoMLSearchStrategy.MapElites</c>. The defaults suit most tabular problems, so start by changing only
/// <see cref="Seed"/> for reproducibility and <see cref="ComplexityBinCount"/> for a coarser or finer archive.</para>
/// <para>
/// The algorithm follows Mouret and Clune, "Illuminating search spaces by mapping elites" (2015). Proposal
/// randomness is derived from <see cref="Seed"/> through stable per-candidate streams rather than a shared
/// global generator, so the search is reproducible whenever the underlying model training is, and duplicate
/// specifications are rejected before training so they never consume the trial budget.
/// </para>
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

    /// <summary>Gets or sets the number of independent island archives; one, the default, keeps a single population.</summary>
    /// <remarks>
    /// Each island keeps its own archive over the same behaviour grid and proposals are assigned to islands in turn,
    /// so several semi-independent searches run inside one trial budget instead of one. Raise it above one together
    /// with <see cref="MigrationInterval"/>, since islands that never exchange elites simply divide the budget.
    /// </remarks>
    public int IslandCount { get; set; } = 1;

    /// <summary>Gets or sets the committed batches between island migrations; zero, the default, disables migration.</summary>
    /// <remarks>
    /// Migration copies the strongest elites of each island into its neighbour, which is what stops islands from
    /// drifting into unrelated local optima. It does nothing while <see cref="IslandCount"/> is one.
    /// </remarks>
    public int MigrationInterval { get; set; }

    /// <summary>Gets or sets the maximum elites copied out of each source island in one migration round.</summary>
    /// <remarks>The default of two matches the evolution engine's own default, so migration behaves the same way.</remarks>
    public int MigrantsPerIsland { get; set; } = 2;

    internal MapElitesAutoMLOptions SnapshotAndValidate()
    {
        if (InitialPopulationSize <= 0) throw new ArgumentOutOfRangeException(nameof(InitialPopulationSize));
        if (ComplexityBinCount <= 0) throw new ArgumentOutOfRangeException(nameof(ComplexityBinCount));
        if (ArchiveCapacity < 0) throw new ArgumentOutOfRangeException(nameof(ArchiveCapacity));
        ValidateProbability(MutationProbability, nameof(MutationProbability));
        ValidateProbability(ExplorationProbability, nameof(ExplorationProbability));
        if (InspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(InspirationCount));
        if (MaxProposalMultiplier <= 0) throw new ArgumentOutOfRangeException(nameof(MaxProposalMultiplier));
        if (IslandCount <= 0) throw new ArgumentOutOfRangeException(nameof(IslandCount));
        if (MigrationInterval < 0) throw new ArgumentOutOfRangeException(nameof(MigrationInterval));
        if (MigrantsPerIsland <= 0) throw new ArgumentOutOfRangeException(nameof(MigrantsPerIsland));

        return new MapElitesAutoMLOptions
        {
            Seed = Seed,
            InitialPopulationSize = InitialPopulationSize,
            ComplexityBinCount = ComplexityBinCount,
            ArchiveCapacity = ArchiveCapacity,
            MutationProbability = MutationProbability,
            ExplorationProbability = ExplorationProbability,
            InspirationCount = InspirationCount,
            MaxProposalMultiplier = MaxProposalMultiplier,
            IslandCount = IslandCount,
            MigrationInterval = MigrationInterval,
            MigrantsPerIsland = MigrantsPerIsland
        };
    }

    private static void ValidateProbability(double value, string parameterName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0 || value > 1)
            throw new ArgumentOutOfRangeException(parameterName, "Probabilities must be finite values in [0, 1].");
    }
}
