using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the Band-Split RNN enhancement model.
/// </summary>
/// <remarks>
/// <para>
/// Band-Split RNN (Luo &amp; Yu, 2023) splits the spectrogram into non-overlapping frequency
/// bands, processes each band with a shared RNN, then fuses the bands. Originally designed
/// for music source separation, it also excels at speech enhancement by treating noise as
/// a "source" to separate. Band-Split RNN achieves state-of-the-art on both music separation
/// and speech enhancement benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you're in a noisy room trying to hear someone speak.
/// Band-Split RNN works like having multiple specialized listeners, each focused on a
/// different pitch range (bass, midrange, treble). Each listener cleans up their range,
/// and then they combine their results. This divide-and-conquer approach works very well
/// because different types of noise affect different frequency ranges.
/// </para>
/// </remarks>
public class BandSplitRNNEnhancerOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT size.</summary>
    public int FFTSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length.</summary>
    public int HopLength { get; set; } = 128;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small", "medium", or "large").</summary>
    public string Variant { get; set; } = "medium";

    /// <summary>Gets or sets the number of frequency bands.</summary>
    public int NumBands { get; set; } = 24;

    /// <summary>Gets or sets the RNN hidden size per band.</summary>
    public int BandRnnHiddenSize { get; set; } = 128;

    /// <summary>Gets or sets the number of RNN layers per band.</summary>
    public int NumRnnLayers { get; set; } = 6;

    /// <summary>Gets or sets the band fusion hidden dimension.</summary>
    public int FusionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of frequency bins.</summary>
    public int NumFreqBins { get; set; } = 257;

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

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
