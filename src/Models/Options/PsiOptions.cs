namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Private Set Intersection (PSI) protocols.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> PSI lets two or more parties find which IDs they have in common
/// without revealing IDs that aren't shared. This is the first step in vertical federated learning:
/// before parties can jointly train a model, they need to know which entities (patients, customers, etc.)
/// they both have data for.</para>
///
/// <para>Example: Two hospitals want to jointly train a model on shared patients:
/// <code>
/// var psiOptions = new PsiOptions
/// {
///     Protocol = PsiProtocol.DiffieHellman,
///     SecurityParameter = 128,
///     FuzzyMatch = new FuzzyMatchOptions
///     {
///         Strategy = FuzzyMatchStrategy.EditDistance,
///         Threshold = 2
///     }
/// };
/// </code>
/// </para>
/// </remarks>
public class PsiOptions
{
    /// <summary>
    /// Gets or sets the PSI protocol to use for computing the set intersection.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different protocols have different tradeoffs:</para>
    /// <list type="bullet">
    /// <item><description><b>DiffieHellman:</b> Simple, good for small-to-medium sets (up to ~1M items).</description></item>
    /// <item><description><b>ObliviousTransfer:</b> Fastest for large sets, recommended for production.</description></item>
    /// <item><description><b>CircuitBased:</b> Most flexible, needed when computing functions on the intersection.</description></item>
    /// <item><description><b>BloomFilter:</b> Fastest but probabilistic (small false-positive rate).</description></item>
    /// </list>
    /// </remarks>
    public PsiProtocol Protocol { get; set; } = PsiProtocol.DiffieHellman;

    /// <summary>
    /// Gets or sets the cryptographic security parameter in bits.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values mean stronger security but slower computation.
    /// 128-bit is standard for most applications. Use 256-bit for highly sensitive data
    /// or long-term security requirements.</para>
    /// </remarks>
    public int SecurityParameter { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hash function used for element hashing within the protocol.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hash function converts IDs into fixed-size values
    /// for comparison. SHA-256 is the default and suitable for almost all cases.</para>
    /// </remarks>
    public string HashFunction { get; set; } = "SHA256";

    /// <summary>
    /// Gets or sets the maximum expected set size for memory pre-allocation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Setting this helps the protocol allocate memory efficiently.
    /// If your dataset has 100,000 records, set this to 100,000 or slightly higher.
    /// Setting it too low may cause reallocation; setting it too high wastes memory.</para>
    /// </remarks>
    public int MaxSetSize { get; set; } = 1_000_000;

    /// <summary>
    /// Gets or sets the false-positive rate for Bloom filter based PSI.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only used with <see cref="PsiProtocol.BloomFilter"/>.
    /// A false-positive rate of 0.001 means roughly 1 in 1000 non-matching IDs may be
    /// incorrectly reported as matching. Lower rates require more memory.</para>
    /// </remarks>
    public double BloomFilterFalsePositiveRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of hash functions used in Bloom filter PSI.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More hash functions reduce false positives but increase computation.
    /// Set to 0 for automatic optimal selection based on set size and false-positive rate.</para>
    /// </remarks>
    public int BloomFilterHashCount { get; set; }

    /// <summary>
    /// Gets or sets options for fuzzy (approximate) entity matching.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When IDs across parties may have typos, formatting differences,
    /// or other inconsistencies, fuzzy matching finds approximate matches. Set to null for
    /// exact matching only.</para>
    /// </remarks>
    public FuzzyMatchOptions? FuzzyMatch { get; set; }

    /// <summary>
    /// Gets or sets the number of parties participating in the PSI protocol.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard PSI is between two parties. Multi-party PSI (3+)
    /// finds elements common to all parties. For example, 3 hospitals finding patients
    /// that appear in all 3 systems.</para>
    /// </remarks>
    public int NumberOfParties { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to only compute the intersection cardinality (count) without
    /// revealing the actual intersecting elements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sometimes you only need to know how many IDs are shared,
    /// not which ones. Cardinality-only mode is faster and reveals less information.
    /// For VFL training, you typically need the actual intersection, so leave this as false.</para>
    /// </remarks>
    public bool CardinalityOnly { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducible protocol execution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Setting a seed makes the protocol deterministic, which is
    /// useful for testing and debugging. Leave null for cryptographically secure randomness
    /// in production.</para>
    /// </remarks>
    public int? RandomSeed { get; set; }
}
