namespace AiDotNet.Data.Formats;

/// <summary>
/// Configuration options for LMDB-based dataset access.
/// </summary>
public sealed class LmdbDatasetOptions
{
    /// <summary>Path to the LMDB environment directory. Required.</summary>
    public string DataPath { get; set; } = "";
    /// <summary>Maximum size of the LMDB map in bytes. Default is 1 GB.</summary>
    public long MapSize { get; set; } = 1L * 1024 * 1024 * 1024;
    /// <summary>Whether to open the database in read-only mode. Default is true.</summary>
    public bool ReadOnly { get; set; } = true;
    /// <summary>Maximum number of readers. Default is 128.</summary>
    public int MaxReaders { get; set; } = 128;
}
