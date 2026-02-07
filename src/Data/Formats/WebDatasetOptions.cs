namespace AiDotNet.Data.Formats;

/// <summary>
/// Configuration options for the WebDataset loader.
/// </summary>
public sealed class WebDatasetOptions
{
    /// <summary>Whether to shuffle samples after reading. Default is true.</summary>
    public bool Shuffle { get; set; } = true;
    /// <summary>Buffer size for shuffle (number of samples to buffer before shuffling). Default is 1000.</summary>
    public int ShuffleBufferSize { get; set; } = 1000;
    /// <summary>Optional maximum number of samples to read.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Optional random seed for reproducible shuffling.</summary>
    public int? Seed { get; set; }
    /// <summary>File extensions to include as data fields (e.g., ".jpg", ".txt", ".json"). Null means all.</summary>
    public HashSet<string>? IncludeExtensions { get; set; }
}
