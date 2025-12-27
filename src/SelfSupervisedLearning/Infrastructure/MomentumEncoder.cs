using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;

namespace AiDotNet.SelfSupervisedLearning.Infrastructure;

/// <summary>
/// Momentum-updated encoder for self-supervised learning methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A momentum encoder is a copy of the main encoder that updates
/// more slowly using exponential moving average (EMA). This provides stable, consistent targets
/// during self-supervised training.</para>
///
/// <para><b>Update formula:</b></para>
/// <code>
/// θ_momentum = m * θ_momentum + (1 - m) * θ_main
/// </code>
/// <para>Where m is momentum (typically 0.99-0.9999).</para>
///
/// <para><b>Why slow updates?</b></para>
/// <list type="bullet">
/// <item>Provides stable targets that don't change rapidly</item>
/// <item>Prevents collapse in methods like BYOL</item>
/// <item>Ensures consistent embeddings in memory bank (MoCo)</item>
/// </list>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// // Initialize with copy of main encoder
/// var momentumEncoder = new MomentumEncoder&lt;float&gt;(mainEncoder.Clone(), momentum: 0.999);
///
/// // Training loop:
/// var targets = momentumEncoder.Encode(augmentedBatch2);  // Get targets
/// // ... compute loss with main encoder output ...
/// momentumEncoder.UpdateFromMainEncoder(mainEncoder);  // EMA update
/// </code>
/// </remarks>
public class MomentumEncoder<T> : IMomentumEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T> _encoder;
    private double _momentum;

    /// <inheritdoc />
    public double Momentum => _momentum;

    /// <inheritdoc />
    public INeuralNetwork<T> Encoder => _encoder;

    /// <summary>
    /// Initializes a new instance of the MomentumEncoder class.
    /// </summary>
    /// <param name="encoder">The encoder network (should be a copy/clone of the main encoder).</param>
    /// <param name="momentum">Initial momentum coefficient (0-1, typically 0.99-0.9999).</param>
    public MomentumEncoder(INeuralNetwork<T> encoder, double momentum = 0.999)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));

        if (momentum < 0 || momentum > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be between 0 and 1");
        }

        _momentum = momentum;

        // Set to evaluation mode by default (no dropout, etc.)
        _encoder.SetTrainingMode(false);
    }

    /// <inheritdoc />
    public Tensor<T> Encode(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Always use evaluation mode for momentum encoder
        _encoder.SetTrainingMode(false);

        // Forward pass without storing gradients
        return _encoder.Predict(input);
    }

    /// <inheritdoc />
    public void UpdateFromMainEncoder(INeuralNetwork<T> mainEncoder)
    {
        if (mainEncoder is null) throw new ArgumentNullException(nameof(mainEncoder));

        var mainParams = mainEncoder.GetParameters();
        UpdateFromParameters(mainParams);
    }

    /// <inheritdoc />
    public void UpdateFromParameters(Vector<T> mainEncoderParams)
    {
        if (mainEncoderParams is null) throw new ArgumentNullException(nameof(mainEncoderParams));

        var momentumParams = _encoder.GetParameters();

        if (mainEncoderParams.Length != momentumParams.Length)
        {
            throw new ArgumentException(
                $"Parameter count mismatch. Main: {mainEncoderParams.Length}, Momentum: {momentumParams.Length}",
                nameof(mainEncoderParams));
        }

        // EMA update: θ_momentum = m * θ_momentum + (1 - m) * θ_main
        var m = NumOps.FromDouble(_momentum);
        var oneMinusM = NumOps.FromDouble(1.0 - _momentum);

        var newParams = new T[momentumParams.Length];
        for (int i = 0; i < momentumParams.Length; i++)
        {
            newParams[i] = NumOps.Add(
                NumOps.Multiply(m, momentumParams[i]),
                NumOps.Multiply(oneMinusM, mainEncoderParams[i]));
        }

        _encoder.UpdateParameters(new Vector<T>(newParams));
    }

    /// <inheritdoc />
    public void SetMomentum(double momentum)
    {
        if (momentum < 0 || momentum > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be between 0 and 1");
        }

        _momentum = momentum;
    }

    /// <inheritdoc />
    public void CopyFromMainEncoder(INeuralNetwork<T> mainEncoder)
    {
        if (mainEncoder is null) throw new ArgumentNullException(nameof(mainEncoder));

        var mainParams = mainEncoder.GetParameters();
        _encoder.UpdateParameters(mainParams);
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        return _encoder.GetParameters();
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        _encoder.UpdateParameters(parameters);
    }

    /// <summary>
    /// Creates a momentum encoder from a main encoder by cloning.
    /// </summary>
    /// <typeparam name="TEncoder">The specific encoder type.</typeparam>
    /// <param name="mainEncoder">The main encoder to clone.</param>
    /// <param name="momentum">Initial momentum coefficient.</param>
    /// <param name="cloneFunc">Function to clone the encoder.</param>
    /// <returns>A new momentum encoder wrapping a cloned encoder.</returns>
    public static MomentumEncoder<T> Create<TEncoder>(
        TEncoder mainEncoder,
        double momentum,
        Func<TEncoder, TEncoder> cloneFunc) where TEncoder : INeuralNetwork<T>
    {
        if (mainEncoder is null) throw new ArgumentNullException(nameof(mainEncoder));
        if (cloneFunc is null) throw new ArgumentNullException(nameof(cloneFunc));

        var clonedEncoder = cloneFunc(mainEncoder);
        return new MomentumEncoder<T>(clonedEncoder, momentum);
    }

    /// <summary>
    /// Computes the scheduled momentum value based on training progress.
    /// </summary>
    /// <param name="baseMomentum">Starting momentum value.</param>
    /// <param name="finalMomentum">Final momentum value.</param>
    /// <param name="currentEpoch">Current training epoch.</param>
    /// <param name="totalEpochs">Total training epochs.</param>
    /// <returns>The scheduled momentum value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some methods like BYOL schedule momentum to increase
    /// during training. This typically uses a cosine schedule from base to final momentum.</para>
    /// </remarks>
    public static double ScheduleMomentum(
        double baseMomentum,
        double finalMomentum,
        int currentEpoch,
        int totalEpochs)
    {
        if (totalEpochs <= 0) return finalMomentum;

        // Cosine schedule from base to final
        var progress = Math.Min(1.0, (double)currentEpoch / totalEpochs);
        var cosineProgress = (1.0 - Math.Cos(Math.PI * progress)) / 2.0;

        return baseMomentum + (finalMomentum - baseMomentum) * cosineProgress;
    }
}
