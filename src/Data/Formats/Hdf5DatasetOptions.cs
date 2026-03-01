namespace AiDotNet.Data.Formats;

/// <summary>
/// Configuration options for HDF5 dataset access.
/// </summary>
public sealed class Hdf5DatasetOptions
{
    /// <summary>Path to the HDF5 file. Required.</summary>
    public string FilePath { get; set; } = "";
    /// <summary>Name of the features dataset within the file. Default is "features".</summary>
    public string FeaturesDataset { get; set; } = "features";
    /// <summary>Name of the labels dataset within the file. Default is "labels".</summary>
    public string LabelsDataset { get; set; } = "labels";
    /// <summary>Number of samples to read per chunk for buffered I/O. Default is 1000.</summary>
    public int ChunkSize { get; set; } = 1000;
}
