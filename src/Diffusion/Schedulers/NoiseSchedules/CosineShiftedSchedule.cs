using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
        var result = new Vector<T>(numTimesteps);
        double s = 0.008; // Small offset to prevent singularity at t=0

        for (int i = 0; i < numTimesteps; i++)
        {
            double t = (double)i / numTimesteps;
            // Shifted cosine: cos((t + s) / (1 + s) * pi/2)^2, with resolution shift
            double cosArg = (t * _shift + s) / (1.0 + s) * Math.PI / 2.0;
            double alpha = Math.Cos(cosArg);
            result[i] = NumOps.FromDouble(Math.Max(0, alpha * alpha));
        }

        return result;
    }
}
