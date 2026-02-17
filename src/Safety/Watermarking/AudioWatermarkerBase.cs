using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Safety.Audio;
using AiDotNet.Safety.Watermarking;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for audio watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for audio watermarkers including strength
/// configuration and spectral utilities. Concrete implementations provide
/// the actual watermarking technique (spread-spectrum, AudioSeal, spectral).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all audio watermarkers.
/// Each watermarker type extends this and adds its own way of embedding invisible
/// signatures in audio content.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class AudioWatermarkerBase<T> : AudioSafetyModuleBase<T>, IAudioWatermarker<T>
{
    /// <summary>
    /// The watermark strength factor (0.0 to 1.0).
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>
    /// Initializes the audio watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Watermark embedding strength. Default: 0.5.</param>
    protected AudioWatermarkerBase(double watermarkStrength = 0.5)
    {
        WatermarkStrength = watermarkStrength;
    }

    /// <inheritdoc />
    public abstract double DetectWatermark(Vector<T> audioSamples, int sampleRate);
}
