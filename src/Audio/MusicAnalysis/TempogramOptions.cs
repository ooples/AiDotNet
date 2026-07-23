using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the neural Tempogram tempo estimation model.
/// </summary>
/// <remarks>
/// <para>
/// The Tempogram model computes a tempo representation over time using a neural approach
/// to onset detection and autocorrelation-based tempo estimation. It provides both global
/// tempo and tempo curves for music with changing tempos.
/// </para>
/// <para>
/// <b>For Beginners:</b> A tempogram shows how the tempo (speed) of music changes over time.
/// This model creates a detailed map of tempo, which is useful for analyzing songs with
/// tempo changes, rubato (expressive timing), or live performances where the tempo varies.
/// </para>
/// </remarks>
public class TempogramOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 512;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the hidden dimension of the onset detector.</summary>
    public int OnsetHiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of onset detector layers.</summary>
    public int NumOnsetLayers { get; set; } = 3;

    /// <summary>Gets or sets the tempo estimation window in frames.</summary>
    public int TempoWindowFrames { get; set; } = 384;

    /// <summary>Gets or sets the minimum BPM to detect.</summary>
    public double MinBPM { get; set; } = 30;

    /// <summary>Gets or sets the maximum BPM to detect.</summary>
    public double MaxBPM { get; set; } = 300;

    /// <summary>Gets or sets the number of tempo bins in the output.</summary>
    public int NumTempoBins { get; set; } = 300;

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
    public double LearningRate { get; set; } = 1e-3;

    #endregion
}
