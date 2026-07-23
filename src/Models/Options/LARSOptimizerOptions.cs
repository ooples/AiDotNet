namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// LARS (Layer-wise Adaptive Rate Scaling) is designed for training with very large batch sizes
/// (4096-32768). It automatically adapts the learning rate for each layer based on the ratio
/// of parameter norm to gradient norm, which helps maintain stable training with large batches.
/// </para>
/// <para><b>For Beginners:</b> When training with large batches (common in self-supervised learning),
/// regular optimizers can become unstable. LARS solves this by automatically adjusting learning rates
/// for each layer based on how "big" the weights and gradients are. This makes training more stable
/// and allows you to use much larger batch sizes, which speeds up training significantly.
/// </para>
/// <para>
/// LARS is particularly important for self-supervised learning methods like SimCLR, which achieve
/// their best results with batch sizes of 4096-8192.
/// </para>
/// <para>
/// Based on the paper "Large Batch Training of Convolutional Networks" by You et al. (2017).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LARSOptimizerOptions&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;
/// {
///     LearningRate = 0.3,        // Base learning rate (will be scaled per-layer)
///     Momentum = 0.9,            // Standard momentum
///     WeightDecay = 1e-4,        // Weight decay
///     TrustCoefficient = 0.001,  // Controls layer-wise LR scaling
///     BatchSize = 4096           // Large batch size for SSL
/// };
/// var optimizer = new LARSOptimizer&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(model, options);
/// </code>
/// </example>
public class LARSOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 4096 for large batch SSL training.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> LARS is specifically designed for large batch sizes.
    /// The default of 4096 is typical for self-supervised learning. You can go up to 32768
    /// with proper learning rate warmup.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the base learning rate for the LARS optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Unlike Adam which typically uses small learning rates (0.001),
    /// LARS uses larger base learning rates (0.1-1.0) because it automatically scales them per layer.
    /// The default of 0.3 works well for most SSL tasks. Use linear scaling: LR = base_lr * batch_size / 256.
    /// </para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the momentum coefficient for the optimizer.
    /// </summary>
    /// <value>The momentum value, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Momentum helps the optimizer maintain direction through
    /// noisy gradients. A value of 0.9 means 90% of the update comes from the previous direction.
    /// Higher values (up to 0.99) can help with very large batches.</para>
    /// </remarks>
    public double Momentum { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the weight decay coefficient.
    /// </summary>
    /// <value>The weight decay coefficient, defaulting to 1e-4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Weight decay prevents the model weights from growing too large.
    /// LARS incorporates weight decay into the layer-wise learning rate calculation, which helps
    /// with numerical stability during large batch training.</para>
    /// </remarks>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the LARS trust coefficient (eta).
    /// </summary>
    /// <value>The trust coefficient, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// The trust coefficient controls how much the layer-wise learning rate scaling affects the update.
    /// A smaller value means more conservative scaling, while a larger value allows larger per-layer
    /// learning rate adjustments.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how aggressively LARS adapts learning rates per layer.
    /// The default of 0.001 is well-tested. Smaller values (0.0001) are more conservative, larger
    /// values (0.01) more aggressive. Stick with the default unless you have specific issues.</para>
    /// </remarks>
    public double TrustCoefficient { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets a small constant added to denominators to prevent division by zero.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a tiny safety value to prevent numerical issues.
    /// You rarely need to change this unless you experience NaN values during training.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets whether to exclude bias parameters and normalization layer parameters from LARS scaling.
    /// </summary>
    /// <value>True to exclude biases from LARS (default), false to apply LARS to all parameters.</value>
    /// <remarks>
    /// <para>
    /// In the original LARS paper and most implementations, bias terms and normalization layer
    /// parameters (BatchNorm, LayerNorm) are excluded from the layer-wise scaling and only use
    /// the base learning rate with momentum.
    /// </para>
    /// <para><b>For Beginners:</b> Some parameters like biases work better with regular learning
    /// rates rather than LARS scaling. Keeping this true (default) follows best practices.</para>
    /// </remarks>
    public bool ExcludeBiasFromLARS { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of warmup steps for learning rate warmup.
    /// </summary>
    /// <value>The number of warmup steps, defaulting to 10 epochs worth of steps.</value>
    /// <remarks>
    /// <para>
    /// Learning rate warmup gradually increases the learning rate from 0 to the target value
    /// over the specified number of steps. This helps stabilize large batch training.
    /// </para>
    /// <para><b>For Beginners:</b> When training with large batches, starting with a full learning
    /// rate can cause training to diverge. Warmup slowly increases the learning rate, giving the
    /// model time to stabilize. A warmup of 10 epochs is typical for SSL.</para>
    /// </remarks>
    public int WarmupEpochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the layer size boundaries for layer-wise scaling.
    /// </summary>
    /// <value>Array of layer sizes that define boundaries between layers for LARS scaling.</value>
    /// <remarks>
    /// <para>
    /// LARS applies different scaling factors to different layers. This array defines the
    /// cumulative sizes of parameters that belong to each layer. If null, each parameter
    /// vector is treated as a single layer.
    /// </para>
    /// <para><b>For Beginners:</b> This tells LARS where one layer ends and another begins
    /// in the flattened parameter vector. If not set, LARS treats all parameters as one layer,
    /// which still works but is less optimal than true layer-wise scaling.</para>
    /// </remarks>
    public int[]? LayerBoundaries { get; set; }

    /// <summary>
    /// Gets or sets which layers should skip LARS scaling and use only the base learning rate.
    /// </summary>
    /// <value>Array of layer indices to skip LARS scaling for.</value>
    /// <remarks>
    /// <para>
    /// Some layers (particularly the final classifier head) may work better without LARS scaling.
    /// This array specifies which layer indices should use only the base learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> In self-supervised learning, you typically want to skip LARS
    /// for the projection head layers that are discarded after pretraining anyway.</para>
    /// </remarks>
    public int[]? SkipLARSLayers { get; set; }

    /// <summary>
    /// Gets or sets whether to use Nesterov momentum instead of standard momentum.
    /// </summary>
    /// <value>True to use Nesterov momentum, false for standard momentum. Default: false</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Nesterov momentum looks ahead before computing gradients,
    /// which can help with convergence. It's slightly more complex but can improve results.
    /// The default (standard momentum) works well for most SSL tasks.</para>
    /// </remarks>
    public bool UseNesterov { get; set; } = false;
}
