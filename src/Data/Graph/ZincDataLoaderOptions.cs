namespace AiDotNet.Data.Graph;

/// <summary>
/// Configuration options for the ZINC molecular dataset data loader.
/// </summary>
/// <remarks>
/// <para>
/// ZINC contains ~250K drug-like molecules from the ZINC database.
/// Standard benchmark for graph regression on constrained solubility.
/// </para>
/// </remarks>
public sealed class ZincDataLoaderOptions
{
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional maximum number of molecules to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Use the 12K subset instead of full 250K. Default is true.</summary>
    public bool UseSubset { get; set; } = true;
}
