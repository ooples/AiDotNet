namespace AiDotNet.Data.Collation;

/// <summary>
/// Defines how individual samples are assembled into a batch.
/// </summary>
/// <typeparam name="TSample">The type of individual samples.</typeparam>
/// <typeparam name="TBatch">The type of the assembled batch.</typeparam>
/// <remarks>
/// <para>
/// Collate functions control how variable-length or heterogeneous samples are combined
/// into a single batch tensor. This is critical for NLP and sequence models where
/// inputs have different lengths.
/// </para>
/// </remarks>
public interface ICollateFunction<in TSample, out TBatch>
{
    /// <summary>
    /// Assembles a collection of individual samples into a batch.
    /// </summary>
    /// <param name="samples">The individual samples to collate.</param>
    /// <returns>The assembled batch.</returns>
    TBatch Collate(IReadOnlyList<TSample> samples);
}
