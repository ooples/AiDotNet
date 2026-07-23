namespace AiDotNet.Interfaces;

/// <summary>
/// Extends <see cref="IDataLoader{T}"/> with checkpoint/resume support for fault-tolerant training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Inspired by PyTorch's StatefulDataLoader (2025), this interface enables mid-epoch
/// checkpointing and exact resumption of data iteration after crashes or preemption.
/// </para>
/// <para><b>For Beginners:</b> If your training crashes halfway through an epoch,
/// a stateful data loader can pick up exactly where it left off instead of starting
/// the epoch over. This saves significant time for large datasets.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("StatefulDataLoader")]
public interface IStatefulDataLoader<T> : IDataLoader<T>
{
    /// <summary>
    /// Captures the current state of the data loader as a serializable object.
    /// </summary>
    /// <returns>A state object that can be serialized and stored.</returns>
    DataLoaderCheckpoint GetState();

    /// <summary>
    /// Restores the data loader to a previously captured state.
    /// </summary>
    /// <param name="state">The state to restore.</param>
    void LoadState(DataLoaderCheckpoint state);
}

/// <summary>
/// Serializable checkpoint for a stateful data loader.
/// </summary>
/// <remarks>
/// <para>
/// Contains all information needed to resume data iteration from an exact position,
/// including the current index, epoch, shuffle order, and RNG state.
/// </para>
/// </remarks>
public class DataLoaderCheckpoint
{
    /// <summary>
    /// The current sample index within the epoch.
    /// </summary>
    public int CurrentIndex { get; set; }

    /// <summary>
    /// The current batch index within the epoch.
    /// </summary>
    public int CurrentBatchIndex { get; set; }

    /// <summary>
    /// The current epoch number.
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// The shuffled indices for the current epoch, preserving iteration order.
    /// </summary>
    public int[]? ShuffledIndices { get; set; }

    /// <summary>
    /// The random seed used to generate the current shuffle order.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// The total number of samples in the dataset at checkpoint time.
    /// </summary>
    public int TotalCount { get; set; }

    /// <summary>
    /// The batch size at checkpoint time.
    /// </summary>
    public int BatchSize { get; set; }

    /// <summary>
    /// UTC timestamp when the checkpoint was created.
    /// </summary>
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Optional metadata for custom state information.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
