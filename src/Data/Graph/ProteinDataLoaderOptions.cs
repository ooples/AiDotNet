namespace AiDotNet.Data.Graph;

/// <summary>
/// Configuration options for the protein structure graph data loader.
/// </summary>
/// <remarks>
/// <para>
/// Loads protein structures as graphs where amino acids are nodes and spatial/sequence
/// contacts are edges. Supports function prediction and fold classification tasks.
/// </para>
/// </remarks>
public sealed class ProteinDataLoaderOptions
{
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of proteins to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Contact distance threshold in angstroms. Default is 8.0.</summary>
    public double ContactThreshold { get; set; } = 8.0;
    /// <summary>Number of amino acid feature dimensions. Default is 20 (one-hot encoding).</summary>
    public int FeatureDimension { get; set; } = 20;
    /// <summary>Number of functional classes. Default is 384 (Gene Ontology terms).</summary>
    public int NumClasses { get; set; } = 384;
}
