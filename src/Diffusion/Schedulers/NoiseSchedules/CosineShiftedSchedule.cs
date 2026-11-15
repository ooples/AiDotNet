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
        // Improved-DDPM cosine schedule (Nichol and Dhariwal, 2021, eq. 17) as diffusers' betas_for_alpha_bar builds
        // it: f(t) = cos^2((t + s) / (1 + s) * pi / 2), beta_i = min(1 - f((i + 1) / N) / f(i / N), 0.999) and
        // alpha_cumprod_i = prod_{j <= i} (1 - beta_j), so entry i is the paper's alpha_bar at t = i + 1 (never 1,
        // which would make sigma_0 = 0) and the last beta is clipped as the paper prescribes. The resolution shift
        // follows simple diffusion (Hoogeboom et al., 2023), which divides the SNR by shift^2:
        // alpha_cumprod_shifted = alpha_cumprod / (alpha_cumprod + shift^2 * (1 - alpha_cumprod)).
        // The table is formed in double and rounded to T once, as diffusers forms it in float64: in T (Half) the
        // index ramp stops being exact past 2048 and every intermediate would be rounded again.
        if (numTimesteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(numTimesteps), "The number of timesteps must be positive.");

        var engine = AiDotNetEngine.Current;
        const double s = 0.008; // Small offset to prevent singularity at t=0
        const double maxBeta = 0.999;
        double angleScale = Math.PI / 2.0 / (1.0 + s);
        var t = engine.TensorMultiplyScalar(engine.TensorArange<double>(0.0, numTimesteps + 1, 1.0), 1.0 / numTimesteps);
        var f = engine.TensorSquare(engine.TensorCos(engine.TensorMultiplyScalar(engine.TensorAddScalar(t, s), angleScale)));
        var ratio = engine.TensorDivide(engine.TensorNarrow(f, 0, 1, numTimesteps), engine.TensorNarrow(f, 0, 0, numTimesteps));
        var betas = engine.TensorClamp(engine.ScalarMinusTensor(1.0, ratio), 0.0, maxBeta);
        var alphasCumprod = engine.TensorCumProd(engine.ScalarMinusTensor(1.0, betas), 0);
        var denominator = engine.TensorAdd(alphasCumprod,
            engine.TensorMultiplyScalar(engine.ScalarMinusTensor(1.0, alphasCumprod), _shift * _shift));
        return engine.TensorDivide(alphasCumprod, denominator).Cast<T>().ToVector();
    }
}
