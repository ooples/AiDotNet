using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
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
        var engine = AiDotNetEngine.Current;
        int pixelsPerChannel = noise.Length / numChannels;

        // Offset noise (Guttenberg; diffusers noise_offset): noise + strength * N(0, 1) per channel. The seeded
        // Box-Muller draws stay one per channel (no engine op consumes a caller Random).
        var offsets = new Vector<T>(numChannels);
        for (int c = 0; c < numChannels; c++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            offsets[c] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * _offsetStrength);
        }

        // Every element of channel c gets offset c; elements past numChannels * pixelsPerChannel keep their noise.
        var perElement = engine.TensorBroadcastTo(offsets.AsTensor().Reshape(new[] { numChannels, 1 }),
            new[] { numChannels, pixelsPerChannel }).Reshape(new[] { numChannels * pixelsPerChannel }).ToVector();
        var padded = Vector<T>.Concatenate(perElement, new Vector<T>(noise.Length - perElement.Length));
        return engine.Add(noise, padded);
    }
}
