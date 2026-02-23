namespace AiDotNet.Data.Formats;

/// <summary>
/// Configuration options for the sharded streaming dataset.
/// </summary>
public sealed class ShardedStreamingDatasetOptions
{
    /// <summary>Whether to shuffle shard order each epoch. Default is true.</summary>
    public bool ShuffleShards { get; set; } = true;
    /// <summary>Buffer size for within-shard shuffling. Default is 1000.</summary>
    public int ShuffleBufferSize { get; set; } = 1000;
    /// <summary>Optional maximum number of samples to read per epoch.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Optional random seed for reproducible shuffling.</summary>
    public int? Seed { get; set; }
    /// <summary>
    /// Reserved for future support of parallel shard reading. Currently not used by
    /// <c>ShardedStreamingDataset</c>, and callers should leave this at the default of 1.
    /// </summary>
    public int NumWorkers { get; set; } = 1;
}
