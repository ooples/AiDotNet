using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Implements dynamic loss scaling for mixed-precision training to prevent gradient underflow.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Loss scaling is a technique used in mixed-precision training to prevent very small
/// gradient values from becoming zero (underflow) when using 16-bit precision.
///
/// The problem:
/// - FP16 (Half) can only represent numbers in the range [6e-8, 65504]
/// - During training, gradients are often very small (e.g., 1e-10)
/// - Small gradients underflow to zero in FP16, stopping learning
///
/// The solution:
/// - Scale the loss by a large factor (e.g., 2^16 = 65536) before backpropagation
/// - This makes gradients larger, preventing underflow
/// - Unscale gradients back to their original values before parameter updates
///
/// Dynamic scaling:
/// - Automatically adjusts the scale factor during training
/// - Increases scale when gradients are stable (no overflow)
/// - Decreases scale when gradients overflow (become infinity/NaN)
/// </para>
/// <para><b>Technical Details:</b> The algorithm follows NVIDIA's approach:
/// 1. Start with a large initial scale (default: 2^16 = 65536)
/// 2. If no overflow for N steps, increase scale by growth factor (default: 2.0)
/// 3. If overflow detected, decrease scale by backoff factor (default: 0.5) and skip update
/// 4. Monitor consecutive successful updates for scale adjustment
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a loss scaler with defaults
/// var scaler = new LossScaler&lt;float&gt;(
///     initialScale: 65536.0,
///     dynamicScaling: true
/// );
///
/// // In training loop:
/// float loss = lossFunction.Compute(predictions, targets);
/// float scaledLoss = scaler.ScaleLoss(loss);
///
/// // Backpropagation with scaled loss...
/// var gradients = model.Backward(scaledLoss);
///
/// // Unscale and check for overflow
/// if (scaler.UnscaleGradientsAndCheck(gradients))
/// {
///     // Safe to update parameters
///     optimizer.Update(parameters, gradients);
/// }
/// else
/// {
///     // Skip this update due to gradient overflow
///     Console.WriteLine($"Gradient overflow, scale reduced to {scaler.Scale}");
/// }
/// </code>
/// </example>
public class LossScaler<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Current loss scale factor.
    /// </summary>
    public double Scale { get; private set; }

    /// <summary>
    /// Whether to use dynamic loss scaling.
    /// </summary>
    public bool DynamicScaling { get; set; }

    /// <summary>
    /// Number of consecutive iterations without overflow before increasing scale.
    /// </summary>
    public int GrowthInterval { get; set; }

    /// <summary>
    /// Factor by which to multiply the scale when increasing (default: 2.0).
    /// </summary>
    public double GrowthFactor { get; set; }

    /// <summary>
    /// Factor by which to multiply the scale when decreasing (default: 0.5).
    /// </summary>
    public double BackoffFactor { get; set; }

    /// <summary>
    /// Minimum allowed scale value to prevent excessive reduction.
    /// </summary>
    public double MinScale { get; set; }

    /// <summary>
    /// Maximum allowed scale value to prevent excessive growth.
    /// </summary>
    public double MaxScale { get; set; }

    private int _consecutiveSuccessfulUpdates;
    private int _totalUpdates;
    private int _skippedUpdates;

    /// <summary>
    /// Gets the total number of updates attempted.
    /// </summary>
    public int TotalUpdates => _totalUpdates;

    /// <summary>
    /// Gets the number of updates skipped due to overflow.
    /// </summary>
    public int SkippedUpdates => _skippedUpdates;

    /// <summary>
    /// Gets the overflow rate (skipped / total).
    /// </summary>
    public double OverflowRate => _totalUpdates > 0 ? (double)_skippedUpdates / _totalUpdates : 0.0;

    /// <summary>
    /// Initializes a new instance of the LossScaler class.
    /// </summary>
    /// <param name="initialScale">Initial loss scale factor (default: 65536 = 2^16).</param>
    /// <param name="dynamicScaling">Enable dynamic scale adjustment (default: true).</param>
    /// <param name="growthInterval">Number of successful updates before scaling up (default: 2000).</param>
    /// <param name="growthFactor">Factor to grow scale by (default: 2.0).</param>
    /// <param name="backoffFactor">Factor to reduce scale by (default: 0.5).</param>
    /// <param name="minScale">Minimum scale value (default: 1.0).</param>
    /// <param name="maxScale">Maximum scale value (default: 2^24 = 16777216).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Default values follow NVIDIA's mixed-precision training recommendations:
    /// - Initial scale of 2^16 works well for most models
    /// - Growth interval of 2000 prevents oscillation
    /// - Growth factor of 2.0 and backoff of 0.5 balance exploration
    /// - Min/max bounds prevent extreme scale values
    /// </para>
    /// </remarks>
    public LossScaler(
        double initialScale = 65536.0,
        bool dynamicScaling = true,
        int growthInterval = 2000,
        double growthFactor = 2.0,
        double backoffFactor = 0.5,
        double minScale = 1.0,
        double maxScale = 16777216.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        Scale = initialScale;
        DynamicScaling = dynamicScaling;
        GrowthInterval = growthInterval;
        GrowthFactor = growthFactor;
        BackoffFactor = backoffFactor;
        MinScale = minScale;
        MaxScale = maxScale;
        _consecutiveSuccessfulUpdates = 0;
        _totalUpdates = 0;
        _skippedUpdates = 0;
    }

    /// <summary>
    /// Scales the loss value to prevent gradient underflow.
    /// </summary>
    /// <param name="loss">The original loss value.</param>
    /// <returns>The scaled loss value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies your loss by the scale factor.
    /// The scaled loss is used for backpropagation, which makes all gradients proportionally larger.
    /// </para>
    /// </remarks>
    public T ScaleLoss(T loss)
    {
        T scaleValue = _numOps.FromDouble(Scale);
        return _numOps.Multiply(loss, scaleValue);
    }

    /// <summary>
    /// Unscales a single gradient value.
    /// </summary>
    /// <param name="gradient">The scaled gradient value.</param>
    /// <returns>The unscaled gradient value.</returns>
    public T UnscaleGradient(T gradient)
    {
        T inverseScale = _numOps.FromDouble(1.0 / Scale);
        return _numOps.Multiply(gradient, inverseScale);
    }

    /// <summary>
    /// Unscales all gradients in a tensor.
    /// </summary>
    /// <param name="gradients">The tensor of scaled gradients.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This divides all gradient values by the scale factor,
    /// returning them to their true magnitudes for parameter updates.
    /// </para>
    /// </remarks>
    public void UnscaleGradients(Tensor<T> gradients)
    {
        T inverseScale = _numOps.FromDouble(1.0 / Scale);

        for (int i = 0; i < gradients.Length; i++)
        {
            T scaledValue = gradients._data[i];
            gradients._data[i] = _numOps.Multiply(scaledValue, inverseScale);
        }
    }

    /// <summary>
    /// Unscales all gradients in a vector.
    /// </summary>
    /// <param name="gradients">The vector of scaled gradients.</param>
    public void UnscaleGradients(Vector<T> gradients)
    {
        T inverseScale = _numOps.FromDouble(1.0 / Scale);

        for (int i = 0; i < gradients.Length; i++)
        {
            T scaledValue = gradients[i];
            gradients[i] = _numOps.Multiply(scaledValue, inverseScale);
        }
    }

    /// <summary>
    /// Checks if a single value has overflowed (is NaN or infinity).
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is NaN or infinity; otherwise, false.</returns>
    public bool HasOverflow(T value)
    {
        return _numOps.IsNaN(value) || _numOps.IsInfinity(value);
    }

    /// <summary>
    /// Checks if any gradient in a tensor has overflowed.
    /// </summary>
    /// <param name="gradients">The tensor of gradients to check.</param>
    /// <returns>True if any gradient is NaN or infinity; otherwise, false.</returns>
    public bool DetectOverflow(Tensor<T> gradients)
    {
        for (int i = 0; i < gradients.Length; i++)
        {
            if (HasOverflow(gradients._data[i]))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Checks if any gradient in a vector has overflowed.
    /// </summary>
    /// <param name="gradients">The vector of gradients to check.</param>
    /// <returns>True if any gradient is NaN or infinity; otherwise, false.</returns>
    public bool DetectOverflow(Vector<T> gradients)
    {
        for (int i = 0; i < gradients.Length; i++)
        {
            if (HasOverflow(gradients[i]))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Unscales gradients and checks for overflow, updating the scale factor if dynamic scaling is enabled.
    /// </summary>
    /// <param name="gradients">The tensor of scaled gradients.</param>
    /// <returns>True if gradients are valid and update can proceed; false if overflow detected and update should be skipped.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method to use in your training loop.
    /// It performs three steps:
    /// 1. Unscales the gradients (divides by scale factor)
    /// 2. Checks if any gradients are NaN or infinity
    /// 3. Adjusts the scale factor if dynamic scaling is enabled
    ///
    /// If overflow is detected, you should skip the parameter update for this step.
    /// </para>
    /// </remarks>
    public bool UnscaleGradientsAndCheck(Tensor<T> gradients)
    {
        _totalUpdates++;

        // First unscale the gradients
        UnscaleGradients(gradients);

        // Check for overflow
        bool hasOverflow = DetectOverflow(gradients);

        if (hasOverflow)
        {
            // Overflow detected - reduce scale and skip update
            _skippedUpdates++;
            _consecutiveSuccessfulUpdates = 0;

            if (DynamicScaling)
            {
                Scale = Math.Max(Scale * BackoffFactor, MinScale);
            }

            return false;
        }
        else
        {
            // No overflow - consider increasing scale
            _consecutiveSuccessfulUpdates++;

            if (DynamicScaling && _consecutiveSuccessfulUpdates >= GrowthInterval)
            {
                Scale = Math.Min(Scale * GrowthFactor, MaxScale);
                _consecutiveSuccessfulUpdates = 0;
            }

            return true;
        }
    }

    /// <summary>
    /// Unscales gradients and checks for overflow (vector version).
    /// </summary>
    /// <param name="gradients">The vector of scaled gradients.</param>
    /// <returns>True if gradients are valid; false if overflow detected.</returns>
    public bool UnscaleGradientsAndCheck(Vector<T> gradients)
    {
        _totalUpdates++;

        // First unscale the gradients
        UnscaleGradients(gradients);

        // Check for overflow
        bool hasOverflow = DetectOverflow(gradients);

        if (hasOverflow)
        {
            // Overflow detected - reduce scale and skip update
            _skippedUpdates++;
            _consecutiveSuccessfulUpdates = 0;

            if (DynamicScaling)
            {
                Scale = Math.Max(Scale * BackoffFactor, MinScale);
            }

            return false;
        }
        else
        {
            // No overflow - consider increasing scale
            _consecutiveSuccessfulUpdates++;

            if (DynamicScaling && _consecutiveSuccessfulUpdates >= GrowthInterval)
            {
                Scale = Math.Min(Scale * GrowthFactor, MaxScale);
                _consecutiveSuccessfulUpdates = 0;
            }

            return true;
        }
    }

    /// <summary>
    /// Resets the statistics and scale to initial values.
    /// </summary>
    /// <param name="newInitialScale">Optional new initial scale value.</param>
    public void Reset(double? newInitialScale = null)
    {
        if (newInitialScale.HasValue)
        {
            Scale = newInitialScale.Value;
        }
        _consecutiveSuccessfulUpdates = 0;
        _totalUpdates = 0;
        _skippedUpdates = 0;
    }

    /// <summary>
    /// Gets a summary of the loss scaler's current state.
    /// </summary>
    /// <returns>A string describing the current state.</returns>
    public override string ToString()
    {
        return $"LossScaler<{typeof(T).Name}>: " +
               $"Scale={Scale:F0}, " +
               $"Dynamic={DynamicScaling}, " +
               $"Total Updates={_totalUpdates}, " +
               $"Skipped={_skippedUpdates} ({OverflowRate:P2}), " +
               $"Consecutive Success={_consecutiveSuccessfulUpdates}";
    }
}
