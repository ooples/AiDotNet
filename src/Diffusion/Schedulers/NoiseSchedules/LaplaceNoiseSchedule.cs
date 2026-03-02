using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Schedulers.NoiseSchedules;

/// <summary>
/// Laplace noise schedule using heavy-tailed Laplace distribution for noise sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Replaces Gaussian noise with Laplace-distributed noise during the forward process.
/// The heavier tails of the Laplace distribution help the model handle extreme values
/// better, improving generation of high-contrast and high-frequency details.
/// </para>
/// <para>
/// <b>For Beginners:</b> Laplace noise has more "extreme" values than standard Gaussian
/// noise. Using it during training helps the model handle sharp edges and high-contrast
/// areas better, leading to crisper generated images.
/// </para>
/// </remarks>
public class LaplaceNoiseSchedule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance with the specified Laplace scale parameter.
    /// </summary>
    /// <param name="scale">Scale parameter of the Laplace distribution. Default: 1.0.</param>
    public LaplaceNoiseSchedule(double scale = 1.0)
    {
        _scale = scale;
    }

    /// <summary>
    /// Generates Laplace-distributed noise values.
    /// </summary>
    /// <param name="length">Number of noise values to generate.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>A vector of Laplace-distributed noise values.</returns>
    public Vector<T> SampleNoise(int length, Random random)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            // Inverse CDF of Laplace: x = -b * sign(u) * ln(1 - 2|u|)
            double u = random.NextDouble() - 0.5;
            double sign = u >= 0 ? 1.0 : -1.0;
            double absU = Math.Abs(u);
            double value = -_scale * sign * Math.Log(1.0 - 2.0 * absU + 1e-10);
            result[i] = NumOps.FromDouble(value);
        }
        return result;
    }
}
