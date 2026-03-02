namespace AiDotNet.Data.Graph;

/// <summary>
/// Configuration options for the QM9 molecular property prediction data loader.
/// </summary>
/// <remarks>
/// <para>
/// QM9 contains ~134K small organic molecules with up to 9 heavy atoms (C, H, N, O, F).
/// 19 quantum mechanical properties computed with DFT. Standard benchmark for molecular GNNs.
/// </para>
/// </remarks>
public sealed class Qm9DataLoaderOptions
{
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional maximum number of molecules to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Target property index (0-18) for regression. Default is 0 (dipole moment mu).</summary>
    public int TargetProperty { get; set; }
}
