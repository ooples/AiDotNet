using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Schedulers.NoiseSchedules;

/// <summary>
/// Offset noise schedule that adds a global offset to noise for improved dark/bright image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Standard diffusion noise is zero-mean per pixel, which biases the model toward
/// mid-tone images. Offset noise adds a small per-channel offset (same value for all
/// pixels in a channel), enabling the model to generate very dark or very bright images.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard diffusion models struggle to generate very dark or very
/// bright images because the noise always averages out to medium brightness. Offset noise
/// fixes this by occasionally adding brightness shifts to the entire image during training.
/// </para>
/// <para>
/// Reference: Originally proposed by Nicholas Guttenberg, widely adopted in SD community
/// </para>
/// </remarks>
public class OffsetNoiseSchedule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _offsetStrength;

    /// <summary>
    /// Initializes a new instance with the specified offset strength.
    /// </summary>
    /// <param name="offsetStrength">Strength of the global noise offset. Typical: 0.1. Default: 0.1.</param>
    public OffsetNoiseSchedule(double offsetStrength = 0.1)
    {
        _offsetStrength = offsetStrength;
    }

    /// <summary>
    /// Applies offset noise to a standard Gaussian noise vector.
    /// </summary>
    /// <param name="noise">Standard Gaussian noise vector.</param>
    /// <param name="numChannels">Number of latent channels.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Noise with per-channel offset applied.</returns>
    public Vector<T> ApplyOffset(Vector<T> noise, int numChannels, Random random)
    {
        var result = new Vector<T>(noise.Length);
        int pixelsPerChannel = noise.Length / numChannels;

        for (int c = 0; c < numChannels; c++)
        {
            // Generate per-channel offset
            double offset = (random.NextDouble() * 2.0 - 1.0) * _offsetStrength;
            var offsetT = NumOps.FromDouble(offset);

            int start = c * pixelsPerChannel;
            for (int i = start; i < start + pixelsPerChannel && i < noise.Length; i++)
                result[i] = NumOps.Add(noise[i], offsetT);
        }

        return result;
    }
}
