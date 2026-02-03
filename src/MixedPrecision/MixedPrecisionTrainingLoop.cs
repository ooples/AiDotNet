using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

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

        _network = network ?? throw new ArgumentNullException(nameof(network));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _context = context ?? throw new ArgumentNullException(nameof(context));

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
    /// Performs one training step with mixed-precision.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <returns>True if the step was successful; false if skipped due to gradient overflow.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method performs one complete training iteration:
    /// - Forward pass → Backward pass → Parameter update
    ///
    /// If gradient overflow is detected (gradients become NaN or infinity), the step is skipped
    /// and the loss scale is automatically reduced. This is normal and expected occasionally.
    /// </para>
    /// </remarks>
    public bool TrainStep(Tensor<T> input, Tensor<T> target)
    {
        TotalSteps++;

        Tensor<T> output;

        // Wrap forward pass in MixedPrecisionScope for automatic precision management
        // Layers can check MixedPrecisionScope.Current to determine their precision
        using (var scope = new MixedPrecisionScope(_context, _policy))
        {
            // Step 1: Cast master weights to FP16 working weights
            // This prepares the context for layers that need to access working weights
            _context.CastWeightsToFP16();

            // Step 2: Forward pass within the scope
            // Layers can check MixedPrecisionScope.Current to access:
            // - scope.ShouldUseFP32(layerName) to check if they need full precision
            // - scope.GetFP32Tensor(name) to get FP32 versions of registered tensors
            output = _network.ForwardWithMemory(input);
        }
        // Scope is disposed here, MixedPrecisionScope.Current becomes null

        // Step 3: Compute loss in FP32 (outside scope for numerical stability)
        var outputVector = output.ToVector();
        var targetVector = target.ToVector();
        var loss = _lossFunction.CalculateLoss(outputVector, targetVector);
        LastLoss = loss;

        // Step 4: Get loss scale factor
        float scale = (float)_context.LossScaler.Scale;

        // Step 5: Backward pass (gradients scaled by scale factor)
        // Compute error gradient
        var errorVector = _lossFunction.CalculateDerivative(outputVector, targetVector);

        // Scale the error by the scale factor to prevent gradient underflow
        var scaledError = new Vector<T>(errorVector.Length);
        for (int i = 0; i < errorVector.Length; i++)
        {
            scaledError[i] = (T)(object)((float)(object)errorVector[i]! * scale);
        }

        var errorTensor = Tensor<T>.FromVector(scaledError, output.Shape);

        // Backpropagate with scaled gradients
        _network.Backpropagate(errorTensor);

        // Step 6: Get gradients and unscale them
        var gradients = _network.GetParameterGradients();
        var gradientsFloat = gradients as Vector<float> ?? throw new InvalidOperationException("Gradients must be Vector<float>");

        // Step 7: Unscale and check for overflow
        bool isValid = _context.LossScaler.UnscaleGradientsAndCheck(gradientsFloat);

        if (!isValid)
        {
            // Gradient overflow detected - skip this update
            // The LossScaler has already reduced the scale for the next iteration
            return false;
        }

        // Step 8: Update master weights in FP32
        // Apply gradients using the optimizer
        var parameters = _network.GetParameters();
        var updatedModel = _optimizer.ApplyGradients(parameters, gradients, _network);

        // Update network parameters from master weights
        _network.SetParameters(updatedModel.GetParameters());

        return true;
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
