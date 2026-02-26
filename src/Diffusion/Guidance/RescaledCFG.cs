using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Rescaled Classifier-Free Guidance to prevent over-saturation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Rescales the guided noise prediction to match the standard deviation of the
/// conditional prediction. This prevents the color saturation and contrast
/// blowout that occurs at high CFG scales (e.g., > 10).
/// </para>
/// <para>
/// <b>For Beginners:</b> When you use a very high guidance scale, images can become
/// oversaturated with unnaturally bright colors. This fix automatically adjusts
/// the brightness back to normal while keeping the guidance effect.
/// </para>
/// <para>
/// Reference: Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed", 2023
/// </para>
/// </remarks>
public class RescaledCFG<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _rescaleWeight;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.RescaledCFG;

    /// <summary>
    /// Initializes a new Rescaled CFG instance.
    /// </summary>
    /// <param name="rescaleWeight">Blending weight between standard and rescaled CFG. 0=standard, 1=fully rescaled. Default: 0.7.</param>
    public RescaledCFG(double rescaleWeight = 0.7)
    {
        _rescaleWeight = rescaleWeight;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();

        // Step 1: Standard CFG
        var guided = new Tensor<T>(unconditional.Shape);
        var guidedSpan = guided.AsWritableSpan();
        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < guidedSpan.Length; i++)
        {
            var diff = NumOps.Subtract(condSpan[i], uncondSpan[i]);
            guidedSpan[i] = NumOps.Add(uncondSpan[i], NumOps.Multiply(scaleT, diff));
        }

        // Step 2: Compute standard deviations
        double condStd = ComputeStd(condSpan);
        double guidedStd = ComputeStd(guided.AsSpan());

        if (guidedStd < 1e-8) return guided;

        // Step 3: Rescale to match conditional std
        double rescaleFactor = condStd / guidedStd;
        var rescaleT = NumOps.FromDouble(rescaleFactor);
        var blendT = NumOps.FromDouble(_rescaleWeight);
        var oneMinusBlend = NumOps.FromDouble(1.0 - _rescaleWeight);

        var result = new Tensor<T>(unconditional.Shape);
        var resultSpan = result.AsWritableSpan();
        var guidedReadSpan = guided.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var rescaled = NumOps.Multiply(guidedReadSpan[i], rescaleT);
            // Blend: result = (1 - weight) * guided + weight * rescaled
            resultSpan[i] = NumOps.Add(
                NumOps.Multiply(oneMinusBlend, guidedReadSpan[i]),
                NumOps.Multiply(blendT, rescaled));
        }

        return result;
    }

    private static double ComputeStd(ReadOnlySpan<T> data)
    {
        if (data.Length == 0) return 0;

        double sum = 0;
        double sumSq = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double val = NumOps.ToDouble(data[i]);
            sum += val;
            sumSq += val * val;
        }

        double mean = sum / data.Length;
        double variance = sumSq / data.Length - mean * mean;
        return Math.Sqrt(Math.Max(0, variance));
    }
}
