
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
    /// Gets or sets whether to use per-layer learning rates.
    /// </summary>
    /// <value>
    /// If true, uses different learning rates for different layers. Default is false.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Per-layer learning rates allow different parts of the network
    /// to learn at different speeds, which can improve performance.
    ///
    /// Different layers often need different learning rates:
    /// - Early layers (feature extraction): Usually slower learning rates
    /// - Later layers (task-specific): Usually faster learning rates
    /// - Attention mechanisms: May need different rates than feedforward layers
    ///
    /// When enabled, the learning rate for each layer is computed as:
    /// base_lr * layer_multiplier
    /// where layer_multiplier depends on the layer type and depth.
    /// </para>
    /// </remarks>
    public bool UsePerLayerLearningRates { get; set; } = false;

    /// <summary>
    /// Gets or sets the learning rate multiplier for early layers.
    /// </summary>
    /// <value>
    /// Multiplier applied to base learning rate for early layers. Default is 0.1.
    /// Only used if UsePerLayerLearningRates is true.
    /// </value>
    public T EarlyLayerMultiplier { get; set; } = NumOps.FromDouble(0.1);

    /// <summary>
    /// Gets or sets the learning rate multiplier for middle layers.
    /// </summary>
    /// <value>
    /// Multiplier applied to base learning rate for middle layers. Default is 0.5.
    /// Only used if UsePerLayerLearningRates is true.
    /// </value>
    public T MiddleLayerMultiplier { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// Gets or sets the learning rate multiplier for late layers.
    /// </summary>
    /// <value>
    /// Multiplier applied to base learning rate for late layers. Default is 1.0.
    /// Only used if UsePerLayerLearningRates is true.
    /// </value>
    public T LateLayerMultiplier { get; set; } = NumOps.FromDouble(1.0);

    /// <summary>
    /// Gets or sets the learning rate multiplier for attention layers.
    /// </summary>
    /// <value>
    /// Special multiplier for attention mechanisms. Default is 0.2.
    /// Only used if UsePerLayerLearningRates is true.
    /// </value>
    public T AttentionLayerMultiplier { get; set; } = NumOps.FromDouble(0.2);

    /// <summary>
    /// Gets or sets the learning rate multiplier for normalization layers.
    /// </summary>
    /// <value>
    /// Special multiplier for batch/layer normalization. Default is 0.01.
    /// Only used if UsePerLayerLearningRates is true.
    /// </value>
    public T NormalizationLayerMultiplier { get; set; } = NumOps.FromDouble(0.01);

    /// <summary>
    /// Gets or sets the learning rate decay factor per layer depth.
    /// </summary>
    /// <value>
    /// Decay factor applied to learning rate based on layer depth. Default is 0.95.
    /// Only used if UsePerLayerLearningRates is true.
    /// </value>
    /// <remarks>
    /// Deeper layers get progressively smaller learning rates when enabled.
    /// </remarks>
    public T LayerDepthDecay { get; set; } = NumOps.FromDouble(0.95);

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
    /// <param name="usePerLayerLearningRates">Whether to use different learning rates per layer.</param>
    /// <param name="earlyLayerMultiplier">Learning rate multiplier for early layers.</param>
    /// <param name="middleLayerMultiplier">Learning rate multiplier for middle layers.</param>
    /// <param name="lateLayerMultiplier">Learning rate multiplier for late layers.</param>
    /// <param name="attentionLayerMultiplier">Learning rate multiplier for attention layers.</param>
    /// <param name="normalizationLayerMultiplier">Learning rate multiplier for normalization layers.</param>
    /// <param name="layerDepthDecay">Learning rate decay factor per layer depth.</param>
    public MAMLTrainerConfig(
        double innerLearningRate,
        double metaLearningRate,
        int innerSteps,
        int metaBatchSize = 4,
        int numMetaIterations = 1000,
        bool useFirstOrderApproximation = true,
        bool usePerLayerLearningRates = false,
        double earlyLayerMultiplier = 0.1,
        double middleLayerMultiplier = 0.5,
        double lateLayerMultiplier = 1.0,
        double attentionLayerMultiplier = 0.2,
        double normalizationLayerMultiplier = 0.01,
        double layerDepthDecay = 0.95)
    {
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
        InnerSteps = innerSteps;
        MetaBatchSize = metaBatchSize;
        NumMetaIterations = numMetaIterations;
        UseFirstOrderApproximation = useFirstOrderApproximation;
        UsePerLayerLearningRates = usePerLayerLearningRates;
        EarlyLayerMultiplier = NumOps.FromDouble(earlyLayerMultiplier);
        MiddleLayerMultiplier = NumOps.FromDouble(middleLayerMultiplier);
        LateLayerMultiplier = NumOps.FromDouble(lateLayerMultiplier);
        AttentionLayerMultiplier = NumOps.FromDouble(attentionLayerMultiplier);
        NormalizationLayerMultiplier = NumOps.FromDouble(normalizationLayerMultiplier);
        LayerDepthDecay = NumOps.FromDouble(layerDepthDecay);
    }

    /// <inheritdoc/>
    public bool IsValid()
    {
        var innerLr = Convert.ToDouble(InnerLearningRate);
        var metaLr = Convert.ToDouble(MetaLearningRate);
        var earlyMult = Convert.ToDouble(EarlyLayerMultiplier);
        var middleMult = Convert.ToDouble(MiddleLayerMultiplier);
        var lateMult = Convert.ToDouble(LateLayerMultiplier);
        var attentionMult = Convert.ToDouble(AttentionLayerMultiplier);
        var normMult = Convert.ToDouble(NormalizationLayerMultiplier);
        var depthDecay = Convert.ToDouble(LayerDepthDecay);

        return innerLr > 0 && innerLr <= 1.0 &&
               metaLr > 0 && metaLr <= 1.0 &&
               InnerSteps > 0 && InnerSteps <= 100 &&
               MetaBatchSize > 0 && MetaBatchSize <= 128 &&
               NumMetaIterations > 0 && NumMetaIterations <= 1000000 &&
               earlyMult > 0 && earlyMult <= 10.0 &&
               middleMult > 0 && middleMult <= 10.0 &&
               lateMult > 0 && lateMult <= 10.0 &&
               attentionMult > 0 && attentionMult <= 10.0 &&
               normMult > 0 && normMult <= 10.0 &&
               depthDecay > 0 && depthDecay <= 1.0;
    }
}
