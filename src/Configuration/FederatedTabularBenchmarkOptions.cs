namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for federated tabular benchmark suites.
/// </summary>
/// <remarks>
/// <para>
/// This groups dataset-specific options for tabular benchmarks under a single facade-facing configuration object.
/// </para>
/// <para><b>For Beginners:</b> Tabular benchmarks test models on spreadsheet-like data (rows and columns).
/// </para>
/// </remarks>
public sealed class FederatedTabularBenchmarkOptions
{
    /// <summary>
    /// Gets or sets options for the synthetic non-IID tabular suite.
    /// </summary>
    public SyntheticTabularFederatedBenchmarkOptions? NonIid { get; set; }
}

