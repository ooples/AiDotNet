using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Abstract base class for audio deepfake detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for audio deepfake detectors including sample rate
/// configuration and common spectral analysis utilities. Concrete implementations
/// provide the actual detection algorithm (spectral, voiceprint, watermark).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all audio deepfake
/// detectors. Each detector type extends this and adds its own way of detecting
/// AI-generated or cloned voices.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class AudioDeepfakeDetectorBase<T> : AudioSafetyModuleBase<T>, IAudioDeepfakeDetector<T>
{
    /// <summary>
    /// The default sample rate for audio processing.
    /// </summary>
    protected readonly int DefaultSampleRate;

    /// <summary>
    /// Initializes the audio deepfake detector base.
    /// </summary>
    /// <param name="defaultSampleRate">Default sample rate in Hz. Default: 16000.</param>
    protected AudioDeepfakeDetectorBase(int defaultSampleRate = 16000)
    {
        if (defaultSampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(defaultSampleRate), "Sample rate must be positive.");

        DefaultSampleRate = defaultSampleRate;
    }

    /// <inheritdoc />
    public abstract double GetDeepfakeScore(Vector<T> audioSamples, int sampleRate);
}
