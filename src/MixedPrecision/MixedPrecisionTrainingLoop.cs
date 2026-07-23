using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Implements mixed-precision training loop for neural networks following NVIDIA's approach.
/// </summary>
/// <typeparam name="T">The numeric type (must be float for mixed-precision).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class implements the complete mixed-precision training workflow:
///
/// 1. **Cast weights to FP16** - Convert FP32 master weights to FP16 working weights
/// 2. **Forward pass in FP16** - Fast computation using 16-bit precision
/// 3. **Compute loss in FP32** - Calculate error using 32-bit precision for stability
/// 4. **Scale loss** - Multiply by large factor (e.g., 2^16) to prevent gradient underflow
/// 5. **Backward pass in FP16** - Compute gradients in 16-bit precision
/// 6. **Unscale and cast gradients to FP32** - Convert gradients back to 32-bit and divide by scale
/// 7. **Check for overflow** - Detect NaN/Inf and adjust loss scale if needed
/// 8. **Update master weights in FP32** - Apply gradients to 32-bit master weights
///
/// This workflow provides 2-3x speedup on modern GPUs while maintaining model accuracy.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create training loop
/// var trainLoop = new MixedPrecisionTrainingLoop&lt;float&gt;(
///     network,
///     optimizer,
///     lossFunction,
///     mixedPrecisionContext
/// );
///
/// // Train for one step
/// bool success = trainLoop.TrainStep(inputTensor, targetTensor);
/// if (!success)
/// {
///     Console.WriteLine("Step skipped due to gradient overflow");
/// }
/// </code>
/// </example>
public class MixedPrecisionTrainingLoop<T>
{
    private readonly NeuralNetworkBase<T> _network;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MixedPrecisionContext _context;
    private readonly LayerPrecisionPolicy _policy;

    /// <summary>
    /// Gets the total number of training steps performed.
    /// </summary>
    public int TotalSteps { get; private set; }

    /// <summary>
    /// Gets the number of steps skipped due to gradient overflow.
    /// </summary>
    public int SkippedSteps => _context.LossScaler.SkippedUpdates;

    /// <summary>
    /// Gets the current loss scale factor.
    /// </summary>
    public double CurrentLossScale => _context.LossScaler.Scale;

    /// <summary>
    /// Gets the last computed loss value.
    /// </summary>
    public T? LastLoss { get; private set; }

    /// <summary>
    /// Initializes a new mixed-precision training loop.
    /// </summary>
    /// <param name="network">The neural network to train.</param>
    /// <param name="optimizer">The optimizer to use for parameter updates.</param>
    /// <param name="lossFunction">The loss function to minimize.</param>
    /// <param name="context">The mixed-precision training context.</param>
    /// <param name="policy">The layer precision policy (optional, uses default based on config if null).</param>
    /// <exception cref="ArgumentException">Thrown when T is not float.</exception>
    public MixedPrecisionTrainingLoop(
        NeuralNetworkBase<T> network,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        ILossFunction<T> lossFunction,
        MixedPrecisionContext context,
        LayerPrecisionPolicy? policy = null)
    {
        if (typeof(T) != typeof(float))
        {
            throw new ArgumentException(
                $"Mixed-precision training requires T = float, got T = {typeof(T).Name}");
        }

        Guard.NotNull(network);
        _network = network;
        Guard.NotNull(optimizer);
        _optimizer = optimizer;
        Guard.NotNull(lossFunction);
        _lossFunction = lossFunction;
        Guard.NotNull(context);
        _context = context;

        // Select appropriate policy based on precision type
        _policy = policy ?? GetDefaultPolicy(context.Config.PrecisionType);

        TotalSteps = 0;
    }

    /// <summary>
    /// Gets the default layer precision policy based on the precision type.
    /// </summary>
    private static LayerPrecisionPolicy GetDefaultPolicy(Enums.MixedPrecisionType precisionType)
    {
        return precisionType switch
        {
            Enums.MixedPrecisionType.BF16 => LayerPrecisionPolicy.ForBF16(),
            Enums.MixedPrecisionType.FP8_E4M3 or
            Enums.MixedPrecisionType.FP8_E5M2 or
            Enums.MixedPrecisionType.FP8_Hybrid => LayerPrecisionPolicy.ForFP8(),
            _ => LayerPrecisionPolicy.ForFP16()
        };
    }

    /// <summary>
    /// Gets the layer precision policy used by this training loop.
    /// </summary>
    public LayerPrecisionPolicy Policy => _policy;

    /// <summary>
    /// Gets statistics about the training process.
    /// </summary>
    /// <returns>A string containing training statistics.</returns>
    public string GetStatistics()
    {
        return $"MixedPrecisionTrainingLoop: " +
               $"TotalSteps={TotalSteps}, " +
               $"SkippedSteps={SkippedSteps} ({(double)SkippedSteps / Math.Max(1, TotalSteps):P2}), " +
               $"CurrentScale={CurrentLossScale:F0}, " +
               $"LastLoss={LastLoss}, " +
               $"Policy={_policy.DefaultPrecision}";
    }
}
