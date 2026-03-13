using System.Diagnostics;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// High-level orchestrator for entity alignment in vertical federated learning.
/// </summary>
/// <remarks>
/// <para>Entity alignment is the first step in any vertical FL pipeline. Before parties can
/// jointly train a model, they need to identify which entities (patients, customers, transactions, etc.)
/// exist in all parties' datasets and align their data rows so that features can be correctly paired
/// during training.</para>
///
/// <para><b>For Beginners:</b> Think of entity alignment like matching rows in two spreadsheets
/// that share a common ID column (e.g., patient ID). If Hospital A has patient data in rows 1-1000
/// and Hospital B has patient data in rows 1-500, entity alignment figures out which rows in A
/// correspond to which rows in B, based on shared patient IDs, without either hospital seeing
/// the other's full patient list.</para>
///
/// <para>The <see cref="EntityAligner"/> class provides a convenient facade over the PSI protocols.
/// It handles protocol selection, fuzzy matching, multi-party coordination, and produces
/// alignment mappings ready for use in VFL training.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var aligner = new EntityAligner();
/// var options = new PsiOptions { Protocol = PsiProtocol.DiffieHellman };
///
/// // Two-party alignment
/// var result = aligner.AlignEntities(hospitalA_Ids, hospitalB_Ids, options);
///
/// // Use alignment mappings for VFL training
/// foreach (var (localRow, sharedRow) in result.LocalToSharedIndexMap)
/// {
///     // Pair localRow from party A with the corresponding remote row for training
/// }
/// </code>
///
/// <para><b>Multi-party alignment:</b></para>
/// <code>
/// var partySets = new List&lt;IReadOnlyList&lt;string&gt;&gt; { partyA_Ids, partyB_Ids, partyC_Ids };
/// var multiResult = aligner.AlignMultipleParties(partySets, options);
/// </code>
/// </remarks>
public class EntityAligner
{
    private readonly IPrivateSetIntersection _defaultProtocol;

    /// <summary>
    /// Initializes a new instance of <see cref="EntityAligner"/> with a specified default protocol.
    /// </summary>
    /// <param name="defaultProtocol">The PSI protocol to use when no protocol is specified in options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="defaultProtocol"/> is null.</exception>
    public EntityAligner(IPrivateSetIntersection defaultProtocol)
    {
        _defaultProtocol = defaultProtocol ?? throw new ArgumentNullException(nameof(defaultProtocol));
    }

    /// <summary>
    /// Initializes a new instance of <see cref="EntityAligner"/> with the Diffie-Hellman protocol as default.
    /// </summary>
    public EntityAligner() : this(new DiffieHellmanPsi())
    {
    }

    /// <summary>
    /// Aligns entities between two parties using the configured PSI protocol.
    /// </summary>
    /// <param name="localIds">The local party's entity identifiers.</param>
    /// <param name="remoteIds">The remote party's entity identifiers.</param>
    /// <param name="options">PSI protocol configuration. If null, default options are used.</param>
    /// <returns>An <see cref="EntityAlignmentResult"/> containing the alignment and diagnostics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method for two-party entity alignment.
    /// Pass in both parties' ID lists and get back a result that tells you:</para>
    /// <list type="bullet">
    /// <item><description>Which IDs are shared between the parties.</description></item>
    /// <item><description>How to map each party's local row numbers to shared row numbers for training.</description></item>
    /// <item><description>Overlap statistics to assess data quality.</description></item>
    /// </list>
    /// </remarks>
    public EntityAlignmentResult AlignEntities(
        IReadOnlyList<string> localIds,
        IReadOnlyList<string> remoteIds,
        PsiOptions? options = null)
    {
        if (localIds is null)
        {
            throw new ArgumentNullException(nameof(localIds));
        }

        if (remoteIds is null)
        {
            throw new ArgumentNullException(nameof(remoteIds));
        }

        var effectiveOptions = options ?? new PsiOptions();
        var protocol = SelectProtocol(effectiveOptions);

        var stopwatch = Stopwatch.StartNew();

        PsiResult psiResult;
        if (effectiveOptions.CardinalityOnly)
        {
            int cardinality = protocol.ComputeCardinality(localIds, remoteIds, effectiveOptions);
            psiResult = new PsiResult
            {
                IntersectionSize = cardinality,
                ProtocolUsed = effectiveOptions.Protocol,
                LocalOverlapRatio = localIds.Count > 0 ? (double)cardinality / localIds.Count : 0.0,
                RemoteOverlapRatio = remoteIds.Count > 0 ? (double)cardinality / remoteIds.Count : 0.0
            };
        }
        else
        {
            psiResult = protocol.ComputeIntersection(localIds, remoteIds, effectiveOptions);
        }

        stopwatch.Stop();

        return new EntityAlignmentResult
        {
            PsiResult = psiResult,
            LocalPartySize = localIds.Count,
            RemotePartySize = remoteIds.Count,
            ProtocolUsed = protocol.ProtocolName,
            TotalExecutionTime = stopwatch.Elapsed,
            NumberOfParties = 2,
            IsCardinalityOnly = effectiveOptions.CardinalityOnly,
            IsFuzzyMatch = psiResult.IsFuzzyMatch
        };
    }

    /// <summary>
    /// Computes only the cardinality (count) of the intersection between two parties.
    /// </summary>
    /// <param name="localIds">The local party's entity identifiers.</param>
    /// <param name="remoteIds">The remote party's entity identifiers.</param>
    /// <param name="options">PSI protocol configuration. If null, default options are used.</param>
    /// <returns>The number of shared entities between the two parties.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you only need to know HOW MANY entities
    /// are shared, not which ones. This is useful for deciding whether there's enough overlap
    /// to justify VFL training (e.g., if only 5% overlap, training may not be worthwhile).</para>
    /// </remarks>
    public int ComputeOverlapCount(
        IReadOnlyList<string> localIds,
        IReadOnlyList<string> remoteIds,
        PsiOptions? options = null)
    {
        if (localIds is null)
        {
            throw new ArgumentNullException(nameof(localIds));
        }

        if (remoteIds is null)
        {
            throw new ArgumentNullException(nameof(remoteIds));
        }

        var effectiveOptions = options ?? new PsiOptions();
        var protocol = SelectProtocol(effectiveOptions);
        return protocol.ComputeCardinality(localIds, remoteIds, effectiveOptions);
    }

    /// <summary>
    /// Aligns entities across three or more parties using multi-party PSI.
    /// </summary>
    /// <param name="partySets">A list of ID sets, one per party. The first party acts as the coordinator.</param>
    /// <param name="options">PSI protocol configuration. If null, default options are used.</param>
    /// <returns>An <see cref="EntityAlignmentResult"/> containing the global alignment and diagnostics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When three or more parties want to jointly train a model,
    /// they need to find entities that ALL parties have in common. For example, if Hospital A,
    /// Hospital B, and Hospital C all have data for patient "John", that patient is in the
    /// global intersection. Patients only in two of the three hospitals are excluded.</para>
    ///
    /// <para>The first party in the list acts as the coordinator (leader) who orchestrates
    /// the pairwise comparisons.</para>
    /// </remarks>
    public EntityAlignmentResult AlignMultipleParties(
        IReadOnlyList<IReadOnlyList<string>> partySets,
        PsiOptions? options = null)
    {
        if (partySets is null)
        {
            throw new ArgumentNullException(nameof(partySets));
        }

        if (partySets.Count < 2)
        {
            throw new ArgumentException(
                "Multi-party alignment requires at least 2 parties.", nameof(partySets));
        }

        var effectiveOptions = options ?? new PsiOptions();

        // For two-party case, delegate to the simpler method
        if (partySets.Count == 2)
        {
            return AlignEntities(partySets[0], partySets[1], effectiveOptions);
        }

        var stopwatch = Stopwatch.StartNew();

        // Select the two-party protocol and wrap it in MultiPartyPsi
        var twoPartyProtocol = SelectProtocol(effectiveOptions);
        var multiPartyPsi = new MultiPartyPsi(twoPartyProtocol);
        var multiResult = multiPartyPsi.ComputeMultiPartyIntersection(partySets, effectiveOptions);

        stopwatch.Stop();

        // Convert multi-party result to the standard alignment result format
        // Use party 0 (leader) as the "local" party and build combined mappings
        var localToShared = multiResult.PartyAlignmentMappings.Length > 0
            ? multiResult.PartyAlignmentMappings[0]
            : (IReadOnlyDictionary<int, int>)new Dictionary<int, int>();

        var remoteToShared = multiResult.PartyAlignmentMappings.Length > 1
            ? multiResult.PartyAlignmentMappings[1]
            : (IReadOnlyDictionary<int, int>)new Dictionary<int, int>();

        var psiResult = new PsiResult
        {
            IntersectionIds = multiResult.IntersectionIds,
            IntersectionSize = multiResult.IntersectionSize,
            LocalToSharedIndexMap = localToShared,
            RemoteToSharedIndexMap = remoteToShared,
            ProtocolUsed = effectiveOptions.Protocol,
            LocalOverlapRatio = partySets[0].Count > 0
                ? (double)multiResult.IntersectionSize / partySets[0].Count
                : 0.0,
            RemoteOverlapRatio = partySets.Count > 1 && partySets[1].Count > 0
                ? (double)multiResult.IntersectionSize / partySets[1].Count
                : 0.0
        };

        return new EntityAlignmentResult
        {
            PsiResult = psiResult,
            MultiPartyResult = multiResult,
            LocalPartySize = partySets[0].Count,
            RemotePartySize = partySets.Count > 1 ? partySets[1].Count : 0,
            ProtocolUsed = multiPartyPsi.ProtocolName,
            TotalExecutionTime = stopwatch.Elapsed,
            NumberOfParties = partySets.Count,
            IsCardinalityOnly = false,
            IsFuzzyMatch = psiResult.IsFuzzyMatch
        };
    }

    /// <summary>
    /// Checks whether there is sufficient overlap between two parties' datasets for viable VFL training.
    /// </summary>
    /// <param name="localIds">The local party's entity identifiers.</param>
    /// <param name="remoteIds">The remote party's entity identifiers.</param>
    /// <param name="minimumOverlapRatio">The minimum acceptable overlap ratio (0.0 to 1.0). Default is 0.1 (10%).</param>
    /// <param name="options">PSI protocol configuration. If null, default options are used.</param>
    /// <returns>
    /// A tuple containing: whether the overlap is sufficient, the actual overlap ratio, and the overlap count.
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before investing time and resources into VFL training, it's wise to
    /// check if the parties have enough shared entities. If only 1% of entities are shared,
    /// the model may not have enough aligned data to learn from effectively.</para>
    ///
    /// <para>This method uses cardinality-only PSI, which is faster than full intersection
    /// because it doesn't need to identify which specific entities are shared.</para>
    /// </remarks>
    public (bool IsSufficient, double OverlapRatio, int OverlapCount) CheckOverlapSufficiency(
        IReadOnlyList<string> localIds,
        IReadOnlyList<string> remoteIds,
        double minimumOverlapRatio = 0.1,
        PsiOptions? options = null)
    {
        if (localIds is null)
        {
            throw new ArgumentNullException(nameof(localIds));
        }

        if (remoteIds is null)
        {
            throw new ArgumentNullException(nameof(remoteIds));
        }

        if (minimumOverlapRatio < 0.0 || minimumOverlapRatio > 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(minimumOverlapRatio), "Overlap ratio must be between 0.0 and 1.0.");
        }

        int overlapCount = ComputeOverlapCount(localIds, remoteIds, options);

        // Overlap ratio is relative to the smaller set (worst case)
        int minSetSize = Math.Min(localIds.Count, remoteIds.Count);
        double overlapRatio = minSetSize > 0 ? (double)overlapCount / minSetSize : 0.0;

        return (overlapRatio >= minimumOverlapRatio, overlapRatio, overlapCount);
    }

    /// <summary>
    /// Selects the appropriate PSI protocol implementation based on the options.
    /// </summary>
    /// <remarks>
    /// <para>If the requested protocol matches the default protocol's type, the default is reused.
    /// Otherwise, a new instance of the requested protocol is created.</para>
    /// </remarks>
    private IPrivateSetIntersection SelectProtocol(PsiOptions options)
    {
        // Check if the default protocol matches the requested protocol
        string requestedName = options.Protocol switch
        {
            PsiProtocol.DiffieHellman => "DiffieHellman",
            PsiProtocol.ObliviousTransfer => "ObliviousTransfer",
            PsiProtocol.CircuitBased => "CircuitBased",
            PsiProtocol.BloomFilter => "BloomFilter",
            _ => "DiffieHellman"
        };

        if (string.Equals(_defaultProtocol.ProtocolName, requestedName, StringComparison.Ordinal))
        {
            return _defaultProtocol;
        }

        return options.Protocol switch
        {
            PsiProtocol.DiffieHellman => new DiffieHellmanPsi(),
            PsiProtocol.ObliviousTransfer => new ObliviousTransferPsi(),
            PsiProtocol.CircuitBased => new CircuitBasedPsi(),
            PsiProtocol.BloomFilter => new BloomFilterPsi(),
            _ => _defaultProtocol
        };
    }
}

/// <summary>
/// Contains the results of an entity alignment operation, including the PSI result,
/// party sizes, and diagnostic information.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This wraps the raw PSI result with additional context about
/// the alignment operation, such as how many entities each party started with, which protocol
/// was used, and how long the operation took. This information is useful for logging,
/// monitoring, and deciding whether to proceed with VFL training.</para>
/// </remarks>
public class EntityAlignmentResult
{
    /// <summary>
    /// Gets or sets the underlying PSI result with intersection IDs and alignment mappings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This contains the core output: which IDs are shared and
    /// how to map local row indices to shared indices for VFL training.</para>
    /// </remarks>
    public PsiResult PsiResult { get; set; } = new PsiResult();

    /// <summary>
    /// Gets or sets the multi-party result when more than two parties are involved.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only populated when <see cref="NumberOfParties"/> is 3 or more.
    /// Contains per-party alignment mappings so each party knows how its local rows map
    /// to the shared global indices.</para>
    /// </remarks>
    public MultiPartyPsiResult? MultiPartyResult { get; set; }

    /// <summary>
    /// Gets or sets the number of entities in the local (initiating) party's dataset.
    /// </summary>
    public int LocalPartySize { get; set; }

    /// <summary>
    /// Gets or sets the number of entities in the remote party's dataset.
    /// For multi-party, this is the second party's size.
    /// </summary>
    public int RemotePartySize { get; set; }

    /// <summary>
    /// Gets or sets the name of the PSI protocol used for the alignment.
    /// </summary>
    public string ProtocolUsed { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the total wall-clock time for the alignment operation.
    /// </summary>
    public TimeSpan TotalExecutionTime { get; set; }

    /// <summary>
    /// Gets or sets the number of parties that participated in the alignment.
    /// </summary>
    public int NumberOfParties { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether only the cardinality was computed (no actual intersection IDs).
    /// </summary>
    public bool IsCardinalityOnly { get; set; }

    /// <summary>
    /// Gets or sets whether fuzzy matching was used for entity alignment.
    /// </summary>
    public bool IsFuzzyMatch { get; set; }

    /// <summary>
    /// Gets the number of aligned entities (intersection size).
    /// </summary>
    public int AlignedEntityCount => PsiResult.IntersectionSize;

    /// <summary>
    /// Gets the fraction of the local party's entities that were aligned.
    /// </summary>
    public double LocalAlignmentRate => LocalPartySize > 0
        ? (double)AlignedEntityCount / LocalPartySize
        : 0.0;

    /// <summary>
    /// Gets the fraction of the remote party's entities that were aligned.
    /// </summary>
    public double RemoteAlignmentRate => RemotePartySize > 0
        ? (double)AlignedEntityCount / RemotePartySize
        : 0.0;
}
