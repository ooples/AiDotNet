using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Schedulers.NoiseSchedules;

/// <summary>
/// Cauchy noise schedule using extremely heavy-tailed Cauchy distribution for noise sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Uses Cauchy-distributed noise which has even heavier tails than Laplace. This can
/// improve model robustness to outliers and enhance generation of extreme contrast regions.
/// Should be used carefully as Cauchy distribution has undefined mean and variance.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cauchy noise has the heaviest tails — it produces extreme values
/// more often than Gaussian or Laplace noise. This can help the model generate very
/// sharp, high-contrast images but needs careful tuning.
/// </para>
/// </remarks>
public class CauchyNoiseSchedule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _scale;

    public CauchyNoiseSchedule(double scale = 1.0)
    {
        _scale = scale;
    }

    /// <summary>
    /// Generates Cauchy-distributed noise values.
    /// </summary>
    public Vector<T> SampleNoise(int length, Random random)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            // Inverse CDF of Cauchy: x = gamma * tan(pi * (u - 0.5))
            double u = random.NextDouble();
            u = Math.Max(1e-10, Math.Min(u, 1.0 - 1e-10)); // Avoid tan singularity
            double value = _scale * Math.Tan(Math.PI * (u - 0.5));
            // Clip extreme values for numerical stability
            value = Math.Max(-10, Math.Min(value, 10));
            result[i] = NumOps.FromDouble(value);
        }
        return result;
    }
}
