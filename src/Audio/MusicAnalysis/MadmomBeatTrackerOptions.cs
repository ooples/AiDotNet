using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the Madmom-style neural beat tracker.
/// </summary>
/// <remarks>
/// <para>
/// The Madmom beat tracking system (Bock et al., 2016) uses a recurrent neural network to detect
/// beat positions and downbeat positions in audio. It combines spectrogram features with bidirectional
/// RNNs and a dynamic Bayesian network for beat tracking, achieving state-of-the-art results on
/// multiple beat tracking benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model listens to music and finds exactly where each beat falls,
/// like a musician tapping their foot in time. It can tell you the tempo (beats per minute) and
/// mark every beat position, which is essential for music synchronization, DJ software, and
/// automatic remixing.
/// </para>
/// </remarks>
public class MadmomBeatTrackerOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 441;

    /// <summary>Gets or sets the number of spectrogram bands.</summary>
    public int NumBands { get; set; } = 81;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the RNN hidden size.</summary>
    public int RnnHiddenSize { get; set; } = 256;

    /// <summary>Gets or sets the number of RNN layers.</summary>
    public int NumRnnLayers { get; set; } = 3;

    /// <summary>Gets or sets the peak picking threshold for beat detection.</summary>
    public double PeakThreshold { get; set; } = 0.3;

    /// <summary>Gets or sets the minimum inter-beat interval in seconds.</summary>
    public double MinBeatInterval { get; set; } = 0.2;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.15;

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
