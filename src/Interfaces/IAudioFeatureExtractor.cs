using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for audio feature extraction algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio feature extractors transform raw audio waveforms into meaningful representations
/// that can be used for tasks like speech recognition, music analysis, and audio classification.
/// </para>
/// <para><b>For Beginners:</b> Audio files are just sequences of numbers representing sound waves.
/// Feature extractors convert these raw numbers into more useful formats:
/// <list type="bullet">
/// <item><b>MFCC</b>: Captures how humans perceive different frequencies</item>
/// <item><b>Chroma</b>: Represents musical pitch classes (C, C#, D, etc.)</item>
/// <item><b>Spectral</b>: Measures brightness, contrast, and other spectral properties</item>
/// </list>
/// </para>
/// </remarks>
public interface IAudioFeatureExtractor<T>
{
    /// <summary>
    /// Gets the name of this feature extractor.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the number of features produced per frame.
    /// </summary>
    int FeatureDimension { get; }

    /// <summary>
    /// Gets the sample rate expected by this extractor.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Extracts features from an audio waveform.
    /// </summary>
    /// <param name="audio">The audio waveform as a 1D tensor [samples].</param>
    /// <returns>Features as a 2D tensor [frames, features].</returns>
    Tensor<T> Extract(Tensor<T> audio);

    /// <summary>
    /// Extracts features from an audio waveform.
    /// </summary>
    /// <param name="audio">The audio waveform as a Vector.</param>
    /// <returns>Features as a Matrix [frames x features].</returns>
    Matrix<T> Extract(Vector<T> audio);

    /// <summary>
    /// Extracts features from an audio waveform asynchronously.
    /// </summary>
    /// <param name="audio">The audio waveform.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Features as a 2D tensor [frames, features].</returns>
    Task<Tensor<T>> ExtractAsync(Tensor<T> audio, CancellationToken cancellationToken = default);
}

/// <summary>
/// Options for audio feature extraction.
/// </summary>
public class AudioFeatureOptions
{
    /// <summary>
    /// Gets or sets the sample rate of the audio.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size in samples.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length (stride) between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the window length in samples. Defaults to FftSize.
    /// </summary>
    public int? WindowLength { get; set; }

    /// <summary>
    /// Gets or sets whether to center-pad the signal.
    /// </summary>
    public bool CenterPad { get; set; } = true;

    /// <summary>
    /// Gets the effective window length.
    /// </summary>
    public int EffectiveWindowLength => WindowLength ?? FftSize;
}
