using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for the PeakNetFP spectral peak-based fingerprinting model.
/// </summary>
/// <remarks>
/// <para>
/// PeakNetFP combines traditional spectral peak picking with a neural network for robust audio
/// fingerprinting. It detects spectral peaks in the spectrogram and uses a CNN to encode peak
/// constellations into compact binary hashes, offering both speed and robustness.
/// </para>
/// <para>
/// <b>For Beginners:</b> PeakNetFP identifies songs by finding the "peaks" in their sound
/// spectrum (like the loudest frequencies at each moment) and then uses AI to turn those peaks
/// into a compact code. It's a hybrid approach that combines classical Shazam-like peak picking
/// with modern neural network encoding.
/// </para>
/// </remarks>
public class PeakNetFPOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 8000;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 256;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 256;

    /// <summary>Gets or sets the segment duration in seconds.</summary>
    public double SegmentDurationSec { get; set; } = 1.0;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the fingerprint embedding dimension.</summary>
    public int EmbeddingDim { get; set; } = 128;

    /// <summary>Gets or sets the number of encoder blocks.</summary>
    public int NumEncoderBlocks { get; set; } = 5;

    /// <summary>Gets or sets the base filter count.</summary>
    public int BaseFilters { get; set; } = 32;

    /// <summary>Gets or sets the number of spectral peaks to select per frame.</summary>
    public int PeaksPerFrame { get; set; } = 5;

    /// <summary>Gets or sets the peak neighborhood size for non-max suppression.</summary>
    public int PeakNeighborhood { get; set; } = 10;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Matching

    /// <summary>
    /// Gets or sets the cosine similarity threshold for considering a fingerprint match.
    /// </summary>
    public double MatchThreshold { get; set; } = 0.7;

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

    /// <summary>Gets or sets the contrastive loss temperature.</summary>
    public double Temperature { get; set; } = 0.05;

    #endregion
}
