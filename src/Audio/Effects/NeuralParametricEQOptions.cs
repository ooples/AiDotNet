using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Configuration options for the Neural Parametric EQ model.
/// </summary>
/// <remarks>
/// <para>
/// Neural Parametric EQ (Steinmetz et al., 2022) uses a neural network to automatically
/// estimate parametric EQ settings that match a target frequency response. Instead of
/// manually adjusting filter bands, the model predicts optimal gain, frequency, and Q
/// values for each band. This is useful for automatic audio mastering, hearing aid fitting,
/// and frequency response matching between recordings.
/// </para>
/// <para>
/// <b>For Beginners:</b> A parametric EQ lets you boost or cut specific frequency ranges
/// in audio (more bass, less treble, etc.). Normally you adjust this by ear, but Neural
/// Parametric EQ uses AI to do it automatically. Give it audio and a target sound, and
/// it figures out the right EQ settings.
/// </para>
/// </remarks>
public class NeuralParametricEQOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the FFT size for frequency analysis.</summary>
    public int FFTSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length for STFT.</summary>
    public int HopLength { get; set; } = 512;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small", "base").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of EQ bands to predict.</summary>
    public int NumBands { get; set; } = 6;

    /// <summary>Gets or sets the minimum frequency in Hz for EQ bands.</summary>
    public double MinFrequency { get; set; } = 20.0;

    /// <summary>Gets or sets the maximum frequency in Hz for EQ bands.</summary>
    public double MaxFrequency { get; set; } = 20000.0;

    /// <summary>Gets or sets the gain range in dB (-GainRange to +GainRange).</summary>
    public double GainRange { get; set; } = 12.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
