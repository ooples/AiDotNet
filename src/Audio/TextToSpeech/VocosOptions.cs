using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for the Vocos neural vocoder.
/// </summary>
/// <remarks>
/// <para>
/// Vocos (Siuzdak, 2023, Charactr) replaces the traditional time-domain waveform generation
/// with frequency-domain (ISTFT) reconstruction. Instead of predicting raw audio samples,
/// Vocos predicts the magnitude and phase of the Short-Time Fourier Transform, then applies
/// the inverse STFT to get the waveform. This is more computationally efficient and avoids
/// artifacts common in time-domain vocoders.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most vocoders (like HiFi-GAN) generate audio one sample at a time,
/// which is like painting a picture pixel by pixel. Vocos instead generates audio in the
/// frequency domain - more like describing the colors and patterns mathematically and then
/// rendering the whole image at once. This is:
///
/// - Faster: Fewer computation steps
/// - Cleaner: Avoids "buzzy" artifacts common in time-domain vocoders
/// - More efficient: Smaller model size for similar quality
/// - Versatile: Works well with neural codecs (EnCodec, SoundStream)
/// </para>
/// </remarks>
public class VocosOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 100;

    /// <summary>Gets or sets the FFT size for ISTFT reconstruction.</summary>
    public int FFTSize { get; set; } = 1024;

    /// <summary>Gets or sets the hop length for ISTFT reconstruction.</summary>
    public int HopLength { get; set; } = 256;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("mel", "encodec").</summary>
    /// <remarks>
    /// - "mel": Mel-spectrogram to waveform (standard TTS vocoder)
    /// - "encodec": EnCodec tokens to waveform (neural codec reconstruction)
    /// </remarks>
    public string Variant { get; set; } = "mel";

    /// <summary>Gets or sets the backbone hidden dimension.</summary>
    public int HiddenDim { get; set; } = 512;

    /// <summary>Gets or sets the number of ConvNeXt backbone blocks.</summary>
    public int NumBackboneBlocks { get; set; } = 8;

    /// <summary>Gets or sets the intermediate dimension in ConvNeXt blocks.</summary>
    public int IntermediateDim { get; set; } = 1536;

    /// <summary>Gets or sets the convolution kernel size in backbone blocks.</summary>
    public int ConvKernelSize { get; set; } = 7;

    #endregion

    #region ISTFT Head

    /// <summary>Gets or sets the number of frequency bins for ISTFT output.</summary>
    /// <remarks>
    /// This is typically FFTSize / 2 + 1 (the positive frequency bins).
    /// </remarks>
    public int NumFrequencyBins { get; set; } = 513;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the weight for multi-resolution STFT loss.</summary>
    public double MultiResSTFTLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight for mel-spectrogram loss.</summary>
    public double MelLossWeight { get; set; } = 45.0;

    #endregion
}
