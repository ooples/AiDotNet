using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration for the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a first-order meta-learning algorithm that learns good initialization
/// parameters by repeatedly moving toward task-adapted parameters.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how Reptile learns:
///
/// - <b>InnerLearningRate:</b> How quickly to adapt to each task (typical: 0.01)
/// - <b>MetaLearningRate (Epsilon):</b> How much to move toward adapted parameters (typical: 0.001)
/// - <b>InnerSteps:</b> How many gradient steps per task (typical: 5-10)
/// - <b>MetaBatchSize:</b> For Reptile, typically 1 (online updates)
///
/// Reptile is simpler than MAML:
/// - No second-order derivatives needed
/// - Just averages parameter updates from tasks
/// - Works well with standard optimizers
/// </para>
/// </remarks>
public class ReptileTrainerConfig<T> : IMetaLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public T InnerLearningRate { get; set; } = NumOps.FromDouble(0.01);

    /// <inheritdoc/>
    public T MetaLearningRate { get; set; } = NumOps.FromDouble(0.001);

    /// <inheritdoc/>
    public int InnerSteps { get; set; } = 5;

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 1;

    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Creates a default Reptile configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default values based on the original Reptile paper (Nichol et al., 2018):
    /// - Inner learning rate: 0.01 (conservative for stability)
    /// - Meta learning rate: 0.001 (10x smaller than inner rate)
    /// - Inner steps: 5 (balance between adaptation and speed)
    /// - Meta batch size: 1 (online updates, typical for Reptile)
    /// - Num meta iterations: 1000 (standard training duration)
    /// </remarks>
    public ReptileTrainerConfig()
    {
    }

    /// <summary>
    /// Creates a Reptile configuration with custom values.
    /// </summary>
    /// <param name="innerLearningRate">Learning rate for task adaptation.</param>
    /// <param name="metaLearningRate">Meta-learning rate (epsilon in Reptile).</param>
    /// <param name="innerSteps">Number of gradient steps per task.</param>
    /// <param name="metaBatchSize">Number of tasks per meta-update (typically 1 for Reptile).</param>
    /// <param name="numMetaIterations">Total number of meta-training iterations.</param>
    public ReptileTrainerConfig(double innerLearningRate, double metaLearningRate, int innerSteps, int metaBatchSize = 1, int numMetaIterations = 1000)
    {
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
        InnerSteps = innerSteps;
        MetaBatchSize = metaBatchSize;
        NumMetaIterations = numMetaIterations;
    }

    /// <inheritdoc/>
    public bool IsValid()
    {
        var innerLr = Convert.ToDouble(InnerLearningRate);
        var metaLr = Convert.ToDouble(MetaLearningRate);

        return innerLr > 0 && innerLr <= 1.0 &&
               metaLr > 0 && metaLr <= 1.0 &&
               InnerSteps > 0 && InnerSteps <= 100 &&
               MetaBatchSize > 0 && MetaBatchSize <= 128 &&
               NumMetaIterations > 0 && NumMetaIterations <= 1000000;
    }
}
