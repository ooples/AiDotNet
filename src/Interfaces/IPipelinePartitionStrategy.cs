namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a strategy for partitioning model parameters across pipeline stages.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When splitting a neural network across multiple devices (pipeline parallelism),
/// you need to decide which layers go on which device. This interface defines that decision.
///
/// The default (uniform) strategy just divides parameters evenly, but this can lead to
/// imbalanced workloads because some layers (like attention) are much more expensive than
/// others (like layer normalization). A load-balanced strategy can account for this.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations.</typeparam>
public interface IPipelinePartitionStrategy<T>
{
    /// <summary>
    /// Computes the partition boundaries for the given number of stages.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns an array describing where each stage's parameters
    /// start and how many parameters it owns. For example, with 1000 total parameters and 4 stages,
    /// a uniform partition might return: [(0, 250), (250, 250), (500, 250), (750, 250)].</para>
    /// </remarks>
    /// <param name="totalParameters">Total number of parameters in the model.</param>
    /// <param name="numStages">Number of pipeline stages to partition across.</param>
    /// <returns>
    /// An array of (startIndex, size) tuples, one per stage, describing each stage's
    /// parameter shard boundaries.
    /// </returns>
    (int StartIndex, int Size)[] ComputePartition(int totalParameters, int numStages);
}
