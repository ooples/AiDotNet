namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for federated text benchmark suites.
/// </summary>
/// <remarks>
/// <para>
/// This groups dataset-specific options for text benchmarks (for example, Sent140 and Shakespeare) under a single
/// facade-facing configuration object.
/// </para>
/// <para><b>For Beginners:</b> Text benchmarks test models on natural language problems. You select a suite (enum)
/// and provide the minimal dataset configuration here.
/// </para>
/// </remarks>
public sealed class FederatedTextBenchmarkOptions
{
    /// <summary>
    /// Gets or sets Sent140 options (LEAF JSON split files).
    /// </summary>
    public Sent140FederatedBenchmarkOptions? Sent140 { get; set; }

    /// <summary>
    /// Gets or sets Shakespeare options (LEAF JSON split files).
    /// </summary>
    public ShakespeareFederatedBenchmarkOptions? Shakespeare { get; set; }

    /// <summary>
    /// Gets or sets Reddit options (LEAF Reddit JSON split files).
    /// </summary>
    public RedditFederatedBenchmarkOptions? Reddit { get; set; }

    /// <summary>
    /// Gets or sets StackOverflow options (token sequence JSON split files).
    /// </summary>
    public StackOverflowFederatedBenchmarkOptions? StackOverflow { get; set; }
}
