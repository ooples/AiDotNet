namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the AdamW optimization algorithm with decoupled weight decay.
/// </summary>
/// <remarks>
/// <para>
/// AdamW (Adam with decoupled Weight decay) differs from Adam with L2 regularization.
/// In Adam with L2, weight decay is applied to the gradient before the adaptive learning rate
/// is computed. In AdamW, weight decay is applied directly to the weights after the Adam update,
/// which has been shown to improve generalization.
/// </para>
/// <para><b>For Beginners:</b> AdamW is an improved version of Adam that handles weight decay (a technique
/// to prevent overfitting) in a mathematically cleaner way. The difference might seem subtle, but AdamW
/// consistently achieves better results than Adam with L2 regularization, especially when training
/// large models like transformers. If you're not sure which to use, AdamW is generally the better choice.
/// </para>
/// <para>
/// Based on the paper "Decoupled Weight Decay Regularization" by Ilya Loshchilov and Frank Hutter.
/// </para>
/// </remarks>
public class AdamWOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for AdamW.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the initial learning rate for the AdamW optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big each step is during training.
    /// AdamW typically uses similar learning rates to Adam (0.001 is a good starting point).
    /// For fine-tuning pre-trained models, smaller values like 2e-5 to 5e-5 are common.</para>
    /// </remarks>
    public new double InitialLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates (momentum).
    /// </summary>
    /// <value>The beta1 value, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta1 controls the momentum of the optimizer. A value of 0.9
    /// means the optimizer gives 90% weight to the previous gradient direction and 10% to the
    /// new gradient. Higher values make updates smoother but potentially slower to adapt.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rate).
    /// </summary>
    /// <value>The beta2 value, defaulting to 0.999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta2 controls how the optimizer adapts the learning rate for each
    /// parameter based on historical gradient magnitudes. The default of 0.999 works well for most cases.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

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
    /// Gets or sets the weight decay coefficient (L2 penalty).
    /// </summary>
    /// <value>The weight decay coefficient, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// Unlike L2 regularization in standard Adam, AdamW applies weight decay directly to the weights,
    /// not through the gradient. This decoupling leads to better generalization.
    /// </para>
    /// <para><b>For Beginners:</b> Weight decay is a regularization technique that prevents the model's
    /// weights from becoming too large, which helps prevent overfitting. A value of 0.01 is a good default.
    /// Increase it if your model overfits (training loss much lower than validation loss), decrease it
    /// if your model underfits (both losses are high).</para>
    /// </remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to apply AMSGrad variant for improved convergence guarantees.
    /// </summary>
    /// <value>True to use AMSGrad variant, false for standard AdamW. Default: false</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> AMSGrad is a modification that maintains the maximum of past
    /// squared gradients rather than an exponential average. This can help in some cases where
    /// standard Adam/AdamW might not converge properly, though in practice the difference is often small.</para>
    /// </remarks>
    public bool UseAMSGrad { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to automatically adjust the Beta parameters during training.
    /// </summary>
    /// <value>True to use adaptive betas (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the algorithm can automatically adjust how much it relies
    /// on past information based on training progress. This can help the optimizer adapt to different
    /// phases of learning.</para>
    /// </remarks>
    public bool UseAdaptiveBetas { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum allowed value for Beta1.
    /// </summary>
    /// <value>The minimum Beta1 value, defaulting to 0.8.</value>
    public double MinBeta1 { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum allowed value for Beta1.
    /// </summary>
    /// <value>The maximum Beta1 value, defaulting to 0.999.</value>
    public double MaxBeta1 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the minimum allowed value for Beta2.
    /// </summary>
    /// <value>The minimum Beta2 value, defaulting to 0.8.</value>
    public double MinBeta2 { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum allowed value for Beta2.
    /// </summary>
    /// <value>The maximum Beta2 value, defaulting to 0.9999.</value>
    public double MaxBeta2 { get; set; } = 0.9999;
}
