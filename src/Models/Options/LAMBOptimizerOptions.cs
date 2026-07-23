namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// LAMB combines Adam's adaptive learning rates (first and second moment estimates) with LARS's
/// layer-wise trust ratio scaling. This enables training with extremely large batch sizes
/// (up to 32K) while maintaining training stability and accuracy.
/// </para>
/// <para><b>For Beginners:</b> LAMB is designed for training large models (like BERT, transformers)
/// with very large batch sizes. It combines:
/// <list type="bullet">
/// <item><b>From Adam:</b> Adaptive learning rates that adjust per-parameter based on gradient history</item>
/// <item><b>From LARS:</b> Layer-wise scaling that stabilizes large batch training</item>
/// </list>
/// The result is an optimizer that can train at batch sizes of 16K-32K while achieving the same
/// accuracy as training with small batches, just much faster.
/// </para>
/// <para>
/// Based on the paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
/// by You et al. (2019).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LAMBOptimizerOptions&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;
/// {
///     LearningRate = 0.00176 * Math.Sqrt(batchSize),  // Square root scaling for LAMB
///     Beta1 = 0.9,
///     Beta2 = 0.999,
///     WeightDecay = 0.01,
///     BatchSize = 8192
/// };
/// var optimizer = new LAMBOptimizer&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(model, options);
/// </code>
/// </example>
public class LAMBOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 8192 for large batch training.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> LAMB is designed for very large batch sizes.
    /// The default of 8192 is typical for BERT/transformer pretraining. You can go up to 32768
    /// with proper learning rate scaling.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 8192;

    /// <summary>
    /// Gets or sets the base learning rate for the LAMB optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Unlike LARS which uses linear scaling, LAMB typically uses
    /// square root scaling: LR = base_lr * sqrt(batch_size / 256). The default of 0.001 is a
    /// good starting point for transformer models.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates (momentum).
    /// </summary>
    /// <value>The beta1 value, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta1 controls the momentum/smoothing of gradient estimates.
    /// A value of 0.9 is standard and works well for most applications.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates.
    /// </summary>
    /// <value>The beta2 value, defaulting to 0.999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta2 controls how the optimizer adapts the learning rate
    /// based on historical gradient magnitudes. The default of 0.999 works well for most cases.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets a small constant added to denominators to prevent division by zero.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a tiny safety value to prevent numerical issues.
    /// LAMB typically uses 1e-6 (slightly larger than Adam's 1e-8) for better stability.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the weight decay coefficient.
    /// </summary>
    /// <value>The weight decay coefficient, defaulting to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Weight decay prevents model weights from growing too large.
    /// LAMB applies decoupled weight decay (like AdamW) for better regularization. A value of
    /// 0.01 is typical for transformer training.</para>
    /// </remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to clip the trust ratio to prevent extreme scaling.
    /// </summary>
    /// <value>True to enable clipping (default), false to disable.</value>
    /// <remarks>
    /// <para>
    /// The trust ratio ||w|| / ||r|| can sometimes become very large, causing instability.
    /// Clipping limits the ratio to [0, max_trust_ratio] for more stable training.
    /// </para>
    /// <para><b>For Beginners:</b> Keeping this enabled (default) prevents training from
    /// becoming unstable when layer weights are much larger than their updates.</para>
    /// </remarks>
    public bool ClipTrustRatio { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum trust ratio when clipping is enabled.
    /// </summary>
    /// <value>The maximum trust ratio, defaulting to 10.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how much the layer-wise scaling can amplify
    /// updates. The default of 10.0 is well-tested for transformer training.</para>
    /// </remarks>
    public double MaxTrustRatio { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets whether to exclude bias and normalization parameters from weight decay.
    /// </summary>
    /// <value>True to exclude biases from weight decay (default), false to apply to all.</value>
    /// <remarks>
    /// <para>
    /// Following best practices for transformer training, bias terms and normalization layer
    /// parameters (BatchNorm, LayerNorm) are typically excluded from weight decay.
    /// </para>
    /// <para><b>For Beginners:</b> Bias terms are small and don't benefit from weight decay.
    /// Keeping this true (default) follows established best practices.</para>
    /// </remarks>
    public bool ExcludeBiasFromWeightDecay { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of warmup epochs for learning rate warmup.
    /// </summary>
    /// <value>The number of warmup epochs, defaulting to 1 epoch worth of steps.</value>
    /// <remarks>
    /// <para>
    /// Learning rate warmup gradually increases the learning rate from 0 to the target value.
    /// LAMB typically uses shorter warmup than LARS due to its adaptive nature.
    /// </para>
    /// <para><b>For Beginners:</b> LAMB is more stable than LARS, so it needs less warmup.
    /// 1 epoch of warmup is typically sufficient for most cases.</para>
    /// </remarks>
    public int WarmupEpochs { get; set; } = 1;

    /// <summary>
    /// Gets or sets the layer size boundaries for layer-wise scaling.
    /// </summary>
    /// <value>Array of layer sizes that define boundaries between layers for LAMB scaling.</value>
    /// <remarks>
    /// <para>
    /// LAMB applies different scaling factors to different layers. This array defines the
    /// cumulative sizes of parameters that belong to each layer.
    /// </para>
    /// <para><b>For Beginners:</b> This tells LAMB where one layer ends and another begins.
    /// If not set, all parameters are treated as one layer, which is less optimal.</para>
    /// </remarks>
    public int[]? LayerBoundaries { get; set; }

    /// <summary>
    /// Gets or sets which layers should skip trust ratio scaling and use only Adam updates.
    /// </summary>
    /// <value>Array of layer indices to skip trust ratio scaling for.</value>
    /// <remarks>
    /// <para>
    /// Some layers (particularly embedding layers) may work better without trust ratio scaling.
    /// These layers use only the Adam update without the layer-wise scaling factor.
    /// </para>
    /// <para><b>For Beginners:</b> Embedding layers in transformers often benefit from being
    /// excluded from trust ratio scaling. Set this to skip those layers.</para>
    /// </remarks>
    public int[]? SkipTrustRatioLayers { get; set; }

    /// <summary>
    /// Gets or sets whether to use bias correction for the moment estimates.
    /// </summary>
    /// <value>True to enable bias correction (default), false to disable.</value>
    /// <remarks>
    /// <para>
    /// Bias correction adjusts for the fact that moment estimates are initialized to zero,
    /// which would otherwise cause them to be biased toward zero early in training.
    /// </para>
    /// <para><b>For Beginners:</b> Always keep this enabled (default). It's essential for
    /// correct Adam-style moment estimates, especially at the start of training.</para>
    /// </remarks>
    public bool UseBiasCorrection { get; set; } = true;
}
