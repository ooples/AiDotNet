namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Contains the results of a Private Set Intersection computation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After running PSI, this object tells you:</para>
/// <list type="bullet">
/// <item><description>Which IDs are shared between the parties (the intersection).</description></item>
/// <item><description>How each party's local row index maps to a shared index for aligned training.</description></item>
/// <item><description>Statistics about the intersection (size, overlap ratio, execution time).</description></item>
/// </list>
///
/// <para>The alignment mappings are critical for vertical FL training: they tell each party
/// which of their local data rows correspond to the shared entities so that features from
/// different parties can be correctly paired during training.</para>
/// </remarks>
public class PsiResult
{
    /// <summary>
    /// Gets or sets the intersecting entity IDs found by the PSI protocol.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the IDs that both parties have in common.
    /// For example, if Party A has {Alice, Bob, Charlie} and Party B has {Bob, Charlie, Dave},
    /// this will be {Bob, Charlie}.</para>
    /// </remarks>
    public IReadOnlyList<string> IntersectionIds { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the number of intersecting elements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The count of shared IDs. In cardinality-only mode,
    /// this may be set even when <see cref="IntersectionIds"/> is empty.</para>
    /// </remarks>
    public int IntersectionSize { get; set; }

    /// <summary>
    /// Gets or sets the mapping from local row indices to shared alignment indices for the initiating party.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This dictionary maps "my local data row number" to
    /// "the shared row number used during joint training". For example, if your local row 5
    /// corresponds to shared index 0, training will pair your row 5 with the other party's
    /// corresponding row for shared index 0.</para>
    /// </remarks>
    public IReadOnlyDictionary<int, int> LocalToSharedIndexMap { get; set; } = new Dictionary<int, int>();

    /// <summary>
    /// Gets or sets the mapping from local row indices to shared alignment indices for the remote party.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Same as <see cref="LocalToSharedIndexMap"/> but for the other party.
    /// In a two-party setting, the VFL trainer uses both mappings to align feature vectors.</para>
    /// </remarks>
    public IReadOnlyDictionary<int, int> RemoteToSharedIndexMap { get; set; } = new Dictionary<int, int>();

    /// <summary>
    /// Gets or sets the fraction of the initiating party's IDs that were found in the intersection.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have 1000 IDs and 800 are shared, this is 0.8 (80%).
    /// A low overlap ratio may indicate data quality issues or that the parties serve
    /// very different populations.</para>
    /// </remarks>
    public double LocalOverlapRatio { get; set; }

    /// <summary>
    /// Gets or sets the fraction of the remote party's IDs that were found in the intersection.
    /// </summary>
    public double RemoteOverlapRatio { get; set; }

    /// <summary>
    /// Gets or sets the total time taken to execute the PSI protocol.
    /// </summary>
    public TimeSpan ExecutionTime { get; set; }

    /// <summary>
    /// Gets or sets the PSI protocol that was used.
    /// </summary>
    public Models.Options.PsiProtocol ProtocolUsed { get; set; }

    /// <summary>
    /// Gets or sets whether the result is from a fuzzy (approximate) match.
    /// </summary>
    public bool IsFuzzyMatch { get; set; }

    /// <summary>
    /// Gets or sets confidence scores for fuzzy-matched pairs, keyed by shared index.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using fuzzy matching, each matched pair has a confidence
    /// score between 0.0 and 1.0 indicating how similar the IDs are. Only populated when
    /// fuzzy matching is enabled. Higher scores mean more confident matches.</para>
    /// </remarks>
    public IReadOnlyDictionary<int, double> FuzzyMatchConfidences { get; set; } = new Dictionary<int, double>();
}
