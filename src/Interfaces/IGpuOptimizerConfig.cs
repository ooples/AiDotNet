namespace AiDotNet.Interfaces;

/// <summary>
/// Configuration for GPU-resident optimizer updates.
/// </summary>
/// <remarks>
/// <para>
/// This interface allows layers to receive optimizer-specific configuration
/// for GPU parameter updates. Different optimizer types (SGD, Adam, etc.)
/// have different implementations with their specific hyperparameters.
/// </para>
/// <para><b>For Beginners:</b> When training on GPU, the weights need to be updated
/// using an optimizer (like SGD or Adam). This configuration tells the GPU
/// exactly how to update the weights - with what learning rate, momentum, etc.
/// </para>
/// </remarks>
public interface IGpuOptimizerConfig
{
    /// <summary>
    /// Gets the type of optimizer (SGD, Adam, AdamW, etc.).
    /// </summary>
    GpuOptimizerType OptimizerType { get; }

    /// <summary>
    /// Gets the learning rate for parameter updates.
    /// </summary>
    float LearningRate { get; }

    /// <summary>
    /// Gets the weight decay (L2 regularization) coefficient.
    /// </summary>
    float WeightDecay { get; }

    /// <summary>
    /// Gets the current optimization step (used for bias correction in Adam-family optimizers).
    /// </summary>
    int Step { get; }
}

/// <summary>
/// Enumerates the types of GPU-optimized optimizers available.
/// </summary>
public enum GpuOptimizerType
{
    /// <summary>Simple SGD with optional momentum.</summary>
    Sgd,

    /// <summary>Adam optimizer with adaptive learning rates.</summary>
    Adam,

    /// <summary>AdamW optimizer with decoupled weight decay.</summary>
    AdamW,

    /// <summary>RMSprop optimizer with moving average of squared gradients.</summary>
    RmsProp,

    /// <summary>Adagrad optimizer with accumulated squared gradients.</summary>
    Adagrad,

    /// <summary>Nesterov Accelerated Gradient with lookahead.</summary>
    Nag,

    /// <summary>Layer-wise Adaptive Rate Scaling.</summary>
    Lars,

    /// <summary>Layer-wise Adaptive Moments optimizer.</summary>
    Lamb
}

/// <summary>
/// Configuration for SGD (Stochastic Gradient Descent) optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// SGD updates weights using: w = w - lr * (grad + weightDecay * w) + momentum * velocity
/// </para>
/// <para><b>For Beginners:</b> SGD is the simplest optimizer. It moves weights
/// in the direction opposite to the gradient, scaled by the learning rate.
/// Momentum helps accelerate training by accumulating velocity from past updates.
/// </para>
/// </remarks>
public class SgdGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.Sgd;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the momentum coefficient (typically 0.9).
    /// </summary>
    /// <remarks>
    /// Momentum accumulates past gradients to smooth updates and escape local minima.
    /// Set to 0 for vanilla SGD without momentum.
    /// </remarks>
    public float Momentum { get; init; }

    /// <summary>
    /// Creates a new SGD GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="momentum">Momentum coefficient (default 0.9).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0).</param>
    /// <param name="step">Current optimization step.</param>
    public SgdGpuConfig(float learningRate, float momentum = 0.9f, float weightDecay = 0f, int step = 0)
    {
        LearningRate = learningRate;
        Momentum = momentum;
        WeightDecay = weightDecay;
        Step = step;
    }
}

/// <summary>
/// Configuration for Adam optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// Adam maintains moving averages of gradients (m) and squared gradients (v),
/// with bias correction for the initial steps.
/// </para>
/// <para><b>For Beginners:</b> Adam is one of the most popular optimizers.
/// It adapts the learning rate for each parameter based on:
/// - First moment (mean of gradients) - like momentum
/// - Second moment (variance of gradients) - adapts to gradient magnitude
/// This typically leads to faster convergence than plain SGD.
/// </para>
/// </remarks>
public class AdamGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.Adam;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the exponential decay rate for the first moment estimates (typically 0.9).
    /// </summary>
    public float Beta1 { get; init; }

    /// <summary>
    /// Gets the exponential decay rate for the second moment estimates (typically 0.999).
    /// </summary>
    public float Beta2 { get; init; }

    /// <summary>
    /// Gets the small constant for numerical stability (typically 1e-8).
    /// </summary>
    public float Epsilon { get; init; }

    /// <summary>
    /// Creates a new Adam GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="beta1">Exponential decay rate for first moment (default 0.9).</param>
    /// <param name="beta2">Exponential decay rate for second moment (default 0.999).</param>
    /// <param name="epsilon">Numerical stability constant (default 1e-8).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0).</param>
    /// <param name="step">Current optimization step.</param>
    public AdamGpuConfig(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f,
        float epsilon = 1e-8f, float weightDecay = 0f, int step = 0)
    {
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        Step = step;
    }
}

/// <summary>
/// Configuration for AdamW optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// AdamW is Adam with decoupled weight decay. Instead of adding weight decay to the gradient
/// before the Adam update, it subtracts it directly from the weights after the update.
/// </para>
/// <para><b>For Beginners:</b> AdamW fixes a subtle issue with L2 regularization in Adam.
/// The original Adam with weight decay doesn't properly regularize because the adaptive
/// learning rates interfere. AdamW applies weight decay directly to weights, which works better.
/// </para>
/// </remarks>
public class AdamWGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.AdamW;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the exponential decay rate for the first moment estimates (typically 0.9).
    /// </summary>
    public float Beta1 { get; init; }

    /// <summary>
    /// Gets the exponential decay rate for the second moment estimates (typically 0.999).
    /// </summary>
    public float Beta2 { get; init; }

    /// <summary>
    /// Gets the small constant for numerical stability (typically 1e-8).
    /// </summary>
    public float Epsilon { get; init; }

    /// <summary>
    /// Creates a new AdamW GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="beta1">Exponential decay rate for first moment (default 0.9).</param>
    /// <param name="beta2">Exponential decay rate for second moment (default 0.999).</param>
    /// <param name="epsilon">Numerical stability constant (default 1e-8).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0.01).</param>
    /// <param name="step">Current optimization step.</param>
    public AdamWGpuConfig(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f,
        float epsilon = 1e-8f, float weightDecay = 0.01f, int step = 0)
    {
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        Step = step;
    }
}

/// <summary>
/// Configuration for RMSprop optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// RMSprop maintains a moving average of squared gradients to normalize the gradient.
/// This helps with non-stationary objectives and is particularly useful for RNNs.
/// </para>
/// <para><b>For Beginners:</b> RMSprop adapts the learning rate by dividing by a running
/// average of gradient magnitudes. This helps training be more stable when gradients
/// vary a lot in size - common in recurrent neural networks.
/// </para>
/// </remarks>
public class RmsPropGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.RmsProp;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the decay rate for the moving average (typically 0.9).
    /// </summary>
    public float Rho { get; init; }

    /// <summary>
    /// Gets the small constant for numerical stability (typically 1e-8).
    /// </summary>
    public float Epsilon { get; init; }

    /// <summary>
    /// Creates a new RMSprop GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="rho">Decay rate for moving average (default 0.9).</param>
    /// <param name="epsilon">Numerical stability constant (default 1e-8).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0).</param>
    /// <param name="step">Current optimization step.</param>
    public RmsPropGpuConfig(float learningRate, float rho = 0.9f, float epsilon = 1e-8f,
        float weightDecay = 0f, int step = 0)
    {
        LearningRate = learningRate;
        Rho = rho;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        Step = step;
    }
}

/// <summary>
/// Configuration for Adagrad optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// Adagrad accumulates squared gradients over all time, providing automatic learning rate
/// adaptation. Parameters with frequently occurring features get smaller learning rates.
/// </para>
/// <para><b>For Beginners:</b> Adagrad is good for sparse data because it gives larger
/// updates to infrequent parameters and smaller updates to frequent ones.
/// However, the accumulated squared gradients can make learning rate too small eventually.
/// </para>
/// </remarks>
public class AdagradGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.Adagrad;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the small constant for numerical stability (typically 1e-8).
    /// </summary>
    public float Epsilon { get; init; }

    /// <summary>
    /// Creates a new Adagrad GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="epsilon">Numerical stability constant (default 1e-8).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0).</param>
    /// <param name="step">Current optimization step.</param>
    public AdagradGpuConfig(float learningRate, float epsilon = 1e-8f, float weightDecay = 0f, int step = 0)
    {
        LearningRate = learningRate;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        Step = step;
    }
}

/// <summary>
/// Configuration for Nesterov Accelerated Gradient (NAG) optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// NAG is a variation of momentum that looks ahead by evaluating the gradient
/// at the "lookahead" position. This often leads to better convergence.
/// </para>
/// <para><b>For Beginners:</b> NAG improves on regular momentum by being smarter
/// about where to look. Instead of computing the gradient at the current position,
/// it first moves in the direction of accumulated momentum, then computes the gradient.
/// This "lookahead" helps it slow down before overshooting.
/// </para>
/// </remarks>
public class NagGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.Nag;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the momentum coefficient (typically 0.9).
    /// </summary>
    public float Momentum { get; init; }

    /// <summary>
    /// Creates a new NAG GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="momentum">Momentum coefficient (default 0.9).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0).</param>
    /// <param name="step">Current optimization step.</param>
    public NagGpuConfig(float learningRate, float momentum = 0.9f, float weightDecay = 0f, int step = 0)
    {
        LearningRate = learningRate;
        Momentum = momentum;
        WeightDecay = weightDecay;
        Step = step;
    }
}

/// <summary>
/// Configuration for LARS (Layer-wise Adaptive Rate Scaling) optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// LARS scales the learning rate for each layer based on the ratio of parameter norm
/// to gradient norm. This enables training with very large batch sizes.
/// </para>
/// <para><b>For Beginners:</b> LARS was designed for training with huge batch sizes
/// (like 32K images). It automatically adjusts the learning rate for each layer
/// so that layers with large parameters don't update too fast and layers with
/// small parameters don't update too slow.
/// </para>
/// </remarks>
public class LarsGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.Lars;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the momentum coefficient (typically 0.9).
    /// </summary>
    public float Momentum { get; init; }

    /// <summary>
    /// Gets the trust coefficient for layer-wise scaling (typically 0.001).
    /// </summary>
    public float TrustCoefficient { get; init; }

    /// <summary>
    /// Creates a new LARS GPU configuration.
    /// </summary>
    /// <param name="learningRate">Global learning rate for parameter updates.</param>
    /// <param name="momentum">Momentum coefficient (default 0.9).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0.0001).</param>
    /// <param name="trustCoefficient">Trust coefficient for scaling (default 0.001).</param>
    /// <param name="step">Current optimization step.</param>
    public LarsGpuConfig(float learningRate, float momentum = 0.9f, float weightDecay = 0.0001f,
        float trustCoefficient = 0.001f, int step = 0)
    {
        LearningRate = learningRate;
        Momentum = momentum;
        WeightDecay = weightDecay;
        TrustCoefficient = trustCoefficient;
        Step = step;
    }
}

/// <summary>
/// Configuration for LAMB (Layer-wise Adaptive Moments) optimizer on GPU.
/// </summary>
/// <remarks>
/// <para>
/// LAMB combines Adam's adaptive learning with LARS's layer-wise scaling.
/// It was designed for training BERT and other large transformers with huge batches.
/// </para>
/// <para><b>For Beginners:</b> LAMB is like a combination of Adam and LARS.
/// It uses Adam's moment estimates for adaptive learning AND applies layer-wise
/// scaling like LARS. This enables training very large models (like BERT) with
/// very large batch sizes efficiently.
/// </para>
/// </remarks>
public class LambGpuConfig : IGpuOptimizerConfig
{
    /// <inheritdoc/>
    public GpuOptimizerType OptimizerType => GpuOptimizerType.Lamb;

    /// <inheritdoc/>
    public float LearningRate { get; init; }

    /// <inheritdoc/>
    public float WeightDecay { get; init; }

    /// <inheritdoc/>
    public int Step { get; init; }

    /// <summary>
    /// Gets the exponential decay rate for the first moment estimates (typically 0.9).
    /// </summary>
    public float Beta1 { get; init; }

    /// <summary>
    /// Gets the exponential decay rate for the second moment estimates (typically 0.999).
    /// </summary>
    public float Beta2 { get; init; }

    /// <summary>
    /// Gets the small constant for numerical stability (typically 1e-6).
    /// </summary>
    public float Epsilon { get; init; }

    /// <summary>
    /// Creates a new LAMB GPU configuration.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="beta1">Exponential decay rate for first moment (default 0.9).</param>
    /// <param name="beta2">Exponential decay rate for second moment (default 0.999).</param>
    /// <param name="epsilon">Numerical stability constant (default 1e-6).</param>
    /// <param name="weightDecay">Weight decay coefficient (default 0.01).</param>
    /// <param name="step">Current optimization step.</param>
    public LambGpuConfig(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f,
        float epsilon = 1e-6f, float weightDecay = 0.01f, int step = 0)
    {
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        Step = step;
    }
}
