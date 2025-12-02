
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration for the MAML (Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// MAML is a second-order meta-learning algorithm that learns optimal parameter initializations
/// for rapid adaptation. It differs from Reptile by computing gradients through the adaptation process.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how MAML learns:
///
/// - <b>InnerLearningRate:</b> How quickly to adapt to each task (typical: 0.01)
/// - <b>MetaLearningRate:</b> How much to update meta-parameters (typical: 0.001)
/// - <b>InnerSteps:</b> How many gradient steps per task (typical: 1-5)
/// - <b>MetaBatchSize:</b> Number of tasks per meta-update (typical: 4-32)
/// - <b>UseFirstOrderApproximation:</b> Whether to use FOMAML (faster, typical: true)
///
/// MAML vs Reptile:
/// - MAML computes gradients through adaptation steps (more accurate)
/// - Reptile just averages adapted parameters (simpler)
/// - First-order MAML (FOMAML) approximates MAML for efficiency
/// - Both work well in practice
/// </para>
/// </remarks>
public class MAMLTrainerConfig<T> : IMetaLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public T InnerLearningRate { get; set; } = NumOps.FromDouble(0.01);

    /// <inheritdoc/>
    public T MetaLearningRate { get; set; } = NumOps.FromDouble(0.001);

    /// <inheritdoc/>
    public int InnerSteps { get; set; } = 5;

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;

    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation (FOMAML) instead of full MAML.
    /// </summary>
    /// <value>
    /// If true, ignores second-order derivatives for efficiency. If false, computes full MAML gradients.
    /// Default is true (FOMAML) for better computational efficiency with minimal performance loss.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a speed vs accuracy trade-off:
    ///
    /// <b>First-order MAML (FOMAML, default):</b>
    /// - Faster training (ignores second-order derivatives)
    /// - Uses less memory
    /// - Still very effective in practice
    /// - Recommended for most use cases
    ///
    /// <b>Full MAML:</b>
    /// - More computationally expensive
    /// - Computes gradients through gradient computations
    /// - Theoretically more accurate
    /// - Use only if you need maximum performance
    ///
    /// The original MAML paper (Finn et al., 2017) showed that first-order approximation
    /// performs nearly as well as full MAML while being much faster.
    /// </para>
    /// </remarks>
    public bool UseFirstOrderApproximation { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>
    /// Maximum allowed gradient norm. Gradients exceeding this will be scaled down.
    /// Set to 0 or negative to disable gradient clipping. Default is 10.0.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradient clipping prevents training instability.
    ///
    /// During meta-learning, gradients can sometimes become very large, causing:
    /// - Unstable training (parameters jumping around)
    /// - Numerical overflow (infinity/NaN values)
    /// - Poor convergence
    ///
    /// Gradient clipping limits how large gradients can be:
    /// - If gradients are too large, scale them down proportionally
    /// - This stabilizes training without changing the direction
    /// - Common in production meta-learning systems
    ///
    /// Recommended values:
    /// - 10.0 (default) for most tasks
    /// - 1.0 for very sensitive tasks
    /// - 0 to disable (only if training is already stable)
    /// </para>
    /// </remarks>
    public T MaxGradientNorm { get; set; } = NumOps.FromDouble(10.0);

    /// <summary>
    /// Gets or sets whether to use adaptive meta-learning rates (Adam-style).
    /// </summary>
    /// <value>
    /// If true, uses Adam-style adaptive learning rates for meta-optimization.
    /// If false, uses vanilla SGD. Default is true for better convergence.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adaptive learning rates help training converge faster and more reliably.
    ///
    /// Vanilla SGD uses the same learning rate for all parameters:
    /// - Simple but can be slow to converge
    /// - Sensitive to learning rate choice
    ///
    /// Adam (Adaptive Moment Estimation):
    /// - Adjusts learning rate per parameter
    /// - Remembers recent gradients (momentum)
    /// - More robust to learning rate choice
    /// - Industry standard for deep learning
    /// - Recommended for meta-learning
    ///
    /// This is the meta-optimizer - it updates the meta-parameters, not the task adaptation.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveMetaOptimizer { get; set; } = true;

    /// <summary>
    /// Gets or sets the Adam beta1 parameter for adaptive meta-optimization.
    /// </summary>
    /// <value>
    /// Exponential decay rate for first moment estimates. Default is 0.9.
    /// Only used if UseAdaptiveMetaOptimizer is true.
    /// </value>
    public T AdamBeta1 { get; set; } = NumOps.FromDouble(0.9);

    /// <summary>
    /// Gets or sets the Adam beta2 parameter for adaptive meta-optimization.
    /// </summary>
    /// <value>
    /// Exponential decay rate for second moment estimates. Default is 0.999.
    /// Only used if UseAdaptiveMetaOptimizer is true.
    /// </value>
    public T AdamBeta2 { get; set; } = NumOps.FromDouble(0.999);

    /// <summary>
    /// Gets or sets the Adam epsilon parameter for numerical stability.
    /// </summary>
    /// <value>
    /// Small constant added to denominator for numerical stability. Default is 1e-8.
    /// Only used if UseAdaptiveMetaOptimizer is true.
    /// </value>
    public T AdamEpsilon { get; set; } = NumOps.FromDouble(1e-8);

    /// <summary>
    /// Creates a default MAML configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default values based on the original MAML paper (Finn et al., 2017):
    /// - Inner learning rate: 0.01 (task adaptation rate)
    /// - Meta learning rate: 0.001 (meta-parameter update rate)
    /// - Inner steps: 5 (balance between adaptation quality and speed)
    /// - Meta batch size: 4 (tasks per meta-update, original paper used 2-32)
    /// - Num meta iterations: 1000 (standard training duration)
    /// - Use first-order approximation: true (FOMAML for efficiency)
    /// </remarks>
    public MAMLTrainerConfig()
    {
    }

    /// <summary>
    /// Creates a MAML configuration with custom values.
    /// </summary>
    /// <param name="innerLearningRate">Learning rate for task adaptation.</param>
    /// <param name="metaLearningRate">Meta-learning rate for meta-parameter updates.</param>
    /// <param name="innerSteps">Number of gradient steps per task.</param>
    /// <param name="metaBatchSize">Number of tasks per meta-update.</param>
    /// <param name="numMetaIterations">Total number of meta-training iterations.</param>
    /// <param name="useFirstOrderApproximation">Whether to use FOMAML (true) or full MAML (false).</param>
    public MAMLTrainerConfig(
        double innerLearningRate,
        double metaLearningRate,
        int innerSteps,
        int metaBatchSize = 4,
        int numMetaIterations = 1000,
        bool useFirstOrderApproximation = true)
    {
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
        InnerSteps = innerSteps;
        MetaBatchSize = metaBatchSize;
        NumMetaIterations = numMetaIterations;
        UseFirstOrderApproximation = useFirstOrderApproximation;
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
