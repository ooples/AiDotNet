using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Divides model parameters evenly across pipeline stages.
/// </summary>
/// <remarks>
/// <para>
/// This is the simplest partitioning strategy: each stage gets approximately the same
/// number of parameters. When the total isn't evenly divisible, earlier stages get one
/// extra parameter each.
/// </para>
/// <para><b>For Beginners:</b> This is the default strategy. It splits the model like cutting
/// a cake into equal slices. It works well when all layers have similar computational cost,
/// but can cause imbalance when some layers (like attention) are much heavier than others.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations.</typeparam>
public class UniformPartitionStrategy<T> : IPipelinePartitionStrategy<T>
{
    /// <inheritdoc/>
    public (int StartIndex, int Size)[] ComputePartition(int totalParameters, int numStages)
    {
        if (totalParameters <= 0)
        {
            throw new ArgumentException("Total parameters must be positive.", nameof(totalParameters));
        }

        if (numStages <= 0)
        {
            throw new ArgumentException("Number of stages must be positive.", nameof(numStages));
        }

        var partitions = new (int StartIndex, int Size)[numStages];
        int baseSize = totalParameters / numStages;
        int remainder = totalParameters % numStages;
        int currentStart = 0;

        for (int i = 0; i < numStages; i++)
        {
            int size = baseSize + (i < remainder ? 1 : 0);
            partitions[i] = (currentStart, size);
            currentStart += size;
        }

        return partitions;
    }
}
