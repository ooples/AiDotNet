using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers.NoiseSchedules;

/// <summary>
/// Cosine-shifted noise schedule for resolution-adapted diffusion training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Shifts the cosine noise schedule based on image resolution to account for the fact
/// that higher-resolution images need more noise to fully corrupt. The shift factor
/// scales with resolution, ensuring consistent effective noise levels across resolutions.
/// </para>
/// <para>
/// <b>For Beginners:</b> At higher resolutions, the same amount of noise is less disruptive
/// because there are more pixels to average over. This schedule compensates by adding more
/// noise at higher resolutions, so the model trains consistently regardless of image size.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public class CosineShiftedSchedule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _shift;

    /// <summary>
    /// Initializes a new instance with the specified shift factor.
    /// </summary>
    /// <param name="shift">Resolution-dependent shift. Higher for higher resolutions. Default: 1.0.</param>
    public CosineShiftedSchedule(double shift = 1.0)
    {
        _shift = shift;
    }

    /// <summary>
    /// Computes shifted cosine alpha cumulative products.
    /// </summary>
    /// <param name="numTimesteps">Number of training timesteps.</param>
    /// <returns>Alpha cumulative product values with resolution-aware cosine schedule.</returns>
    public Vector<T> ComputeAlphasCumprod(int numTimesteps)
    {
        // Improved-DDPM cosine schedule (Nichol and Dhariwal, 2021): alpha_cumprod(t) = f(t) / f(0) with
        // f(t) = cos^2((t + s) / (1 + s) * pi / 2), t = i / numTimesteps. The resolution shift follows simple
        // diffusion (Hoogeboom et al., 2023), which divides the SNR by shift^2:
        // alpha_cumprod_shifted = alpha_cumprod / (alpha_cumprod + shift^2 * (1 - alpha_cumprod)).
        var engine = AiDotNetEngine.Current;
        double s = 0.008; // Small offset to prevent singularity at t=0
        double angleScale = Math.PI / 2.0 / (1.0 + s);
        var t = engine.TensorMultiplyScalar(engine.TensorArange<T>(0.0, numTimesteps, 1.0), NumOps.FromDouble(1.0 / numTimesteps));
        var angle = engine.TensorMultiplyScalar(engine.TensorAddScalar(t, NumOps.FromDouble(s)), NumOps.FromDouble(angleScale));
        double f0 = Math.Cos(s * angleScale);
        var alphas = engine.TensorMultiplyScalar(engine.TensorSquare(engine.TensorCos(angle)), NumOps.FromDouble(1.0 / (f0 * f0)));
        var shiftSquared = NumOps.FromDouble(_shift * _shift);
        var denominator = engine.TensorAdd(alphas,
            engine.TensorMultiplyScalar(engine.ScalarMinusTensor(NumOps.One, alphas), shiftSquared));
        return engine.TensorDivide(alphas, denominator).ToVector();
    }
}
