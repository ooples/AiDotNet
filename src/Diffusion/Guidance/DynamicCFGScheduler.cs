using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Dynamic Classifier-Free Guidance that adjusts scale per timestep.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Applies a time-dependent guidance scale that starts high for structural coherence
/// in early (noisy) steps and decreases toward the end for fine detail preservation.
/// This reduces over-saturation artifacts common with high static CFG scales.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of using the same guidance strength for every step,
/// this uses stronger guidance at the start (to get the big picture right) and
/// lighter guidance at the end (to keep fine details natural).
/// </para>
/// </remarks>
public class DynamicCFGScheduler<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _minScale;
    private readonly double _maxScale;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.DynamicCFG;

    /// <summary>
    /// Initializes a new Dynamic CFG scheduler.
    /// </summary>
    /// <param name="minScale">Minimum guidance scale at final timestep. Default: 1.0.</param>
    /// <param name="maxScale">Maximum guidance scale at initial timestep. Default: 15.0.</param>
    public DynamicCFGScheduler(double minScale = 1.0, double maxScale = 15.0)
    {
        _minScale = minScale;
        _maxScale = maxScale;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        // Linear interpolation: high at t=1 (noisy), low at t=0 (clean)
        double dynamicScale = _minScale + (_maxScale - _minScale) * timestep;

        var result = new Tensor<T>(unconditional.Shape);
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var scaleT = NumOps.FromDouble(dynamicScale);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var diff = NumOps.Subtract(condSpan[i], uncondSpan[i]);
            resultSpan[i] = NumOps.Add(uncondSpan[i], NumOps.Multiply(scaleT, diff));
        }

        return result;
    }
}
