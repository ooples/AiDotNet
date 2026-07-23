namespace AiDotNet.ActiveLearning;

/// <summary>
/// One pool sample's place in an active-learning ranking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning ranks unlabeled samples by how useful it would be to
/// label them next. This is one sample's entry in that ranking.</para>
/// </remarks>
public sealed class ActiveLearningCandidate
{
    /// <summary>Gets the sample's index in the unlabeled pool.</summary>
    public int PoolIndex { get; init; }

    /// <summary>Gets the strategy's raw informativeness (uncertainty) score for this sample.</summary>
    public double Informativeness { get; init; }

    /// <summary>
    /// Gets the order this sample was chosen into the batch (0 = chosen first), or <c>-1</c> if it was
    /// not selected into the batch.
    /// </summary>
    public int SelectionOrder { get; init; } = -1;

    /// <summary>
    /// Gets the diversity-adjusted score at the moment this sample was chosen
    /// (informativeness minus its redundancy against the already-chosen batch), or the raw
    /// informativeness for unselected samples. This is what makes the batch cover the pool instead of
    /// collecting near-duplicates.
    /// </summary>
    public double MarginalGain { get; init; }
}

/// <summary>
/// The result of an active-learning pass: a full ranking of the unlabeled pool plus the diversity-aware
/// batch chosen for labeling.
/// </summary>
/// <remarks>
/// <para>
/// Beyond a plain uncertainty ranking, the batch is chosen with a diversity/redundancy penalty
/// (BADGE / facility-location style) so it covers the pool rather than collecting many near-identical
/// uncertain samples — the failure mode of naive top-N uncertainty sampling.
/// </para>
/// <para><b>For Beginners:</b> This tells you which unlabeled samples to label next, and why: each
/// sample's uncertainty, whether it made the batch, and the order it was chosen.</para>
/// </remarks>
public sealed class ActiveLearningSelection
{
    /// <summary>Gets the whole pool ranked, most-informative first.</summary>
    public IReadOnlyList<ActiveLearningCandidate> Ranking { get; init; } = System.Array.Empty<ActiveLearningCandidate>();

    /// <summary>Gets the selected sample indices (into the pool), in selection order.</summary>
    public int[] SelectedIndices { get; init; } = System.Array.Empty<int>();

    /// <summary>Gets the requested batch size.</summary>
    public int BatchSize { get; init; }

    /// <summary>Gets the configured strategy's name.</summary>
    public string StrategyName { get; init; } = string.Empty;

    /// <summary>
    /// Gets the space in which diversity was measured: <c>"ModelRepresentation"</c> when a model embedding
    /// was available, otherwise <c>"InputFeatures"</c>.
    /// </summary>
    public string RepresentationSpace { get; init; } = string.Empty;

    /// <summary>
    /// Gets the diversity weight used to penalize redundancy (0 = pure uncertainty, higher = more
    /// diversity pressure).
    /// </summary>
    public double DiversityWeight { get; init; }
}
