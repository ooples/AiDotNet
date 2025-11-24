using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ContinualLearning.Config;

/// <summary>
/// Default configuration for continual learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ContinualLearnerConfig<T> : IContinualLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public T LearningRate { get; set; }

    /// <inheritdoc/>
    public int EpochsPerTask { get; set; }

    /// <inheritdoc/>
    public int BatchSize { get; set; }

    /// <inheritdoc/>
    public int MemorySize { get; set; }

    /// <inheritdoc/>
    public T RegularizationStrength { get; set; }

    /// <summary>
    /// Creates a default configuration with sensible defaults.
    /// </summary>
    public ContinualLearnerConfig()
    {
        LearningRate = NumOps.FromDouble(0.001);
        EpochsPerTask = 10;
        BatchSize = 32;
        MemorySize = 1000;
        RegularizationStrength = NumOps.FromDouble(1000.0);
    }

    /// <summary>
    /// Creates a configuration with specified parameters.
    /// </summary>
    public ContinualLearnerConfig(
        T learningRate,
        int epochsPerTask,
        int batchSize,
        int memorySize,
        T regularizationStrength)
    {
        LearningRate = learningRate;
        EpochsPerTask = epochsPerTask;
        BatchSize = batchSize;
        MemorySize = memorySize;
        RegularizationStrength = regularizationStrength;
    }

    /// <inheritdoc/>
    public bool IsValid()
    {
        return Convert.ToDouble(LearningRate) > 0 &&
               EpochsPerTask > 0 &&
               BatchSize > 0 &&
               MemorySize >= 0 &&
               Convert.ToDouble(RegularizationStrength) >= 0;
    }
}
