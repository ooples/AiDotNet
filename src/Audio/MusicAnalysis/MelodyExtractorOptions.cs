using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the neural Melody Extraction model.
/// </summary>
/// <remarks>
/// <para>
/// The Melody Extractor identifies the primary melodic line from a polyphonic audio recording
/// using a neural network. Unlike pitch detection (which finds any pitch), melody extraction
/// specifically tracks the dominant melody even when other instruments are playing simultaneously.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you listen to a song, you can usually hum along to the main
/// melody even though many instruments are playing. This model does the same thing—it finds
/// and extracts just the main tune from a full song, ignoring background harmonies and rhythms.
/// </para>
/// </remarks>
public class MelodyExtractorOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 256;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of pitch bins in the output.</summary>
    public int NumPitchBins { get; set; } = 360;

    /// <summary>Gets or sets the minimum frequency in Hz.</summary>
    public double MinFrequency { get; set; } = 55.0;

    /// <summary>Gets or sets the maximum frequency in Hz.</summary>
    public double MaxFrequency { get; set; } = 1760.0;

    /// <summary>Gets or sets the voicing threshold.</summary>
    public double VoicingThreshold { get; set; } = 0.5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

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

    #endregion
}
