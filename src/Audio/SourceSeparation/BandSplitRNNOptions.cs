using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Configuration options for the BandSplitRNN source separation model.
/// </summary>
/// <remarks>
/// <para>
/// BandSplitRNN (Luo and Yu, 2023) is the original Band-Split RNN model designed specifically for
/// music source separation. It splits the spectrogram into non-overlapping frequency bands, processes
/// each with a shared band-level RNN, applies cross-band fusion via a sequence-level RNN, and then
/// reconstructs source-specific masks. It achieves 10.0+ dB SDR on MUSDB18-HQ.
/// </para>
/// <para>
/// <b>For Beginners:</b> BandSplitRNN works like a team of specialists: each specialist listens to a
/// specific frequency range (e.g., bass frequencies, mid-range, treble), learns to identify what
/// belongs to each instrument in their range, and then they all compare notes to produce a consistent
/// separation across the full frequency spectrum. This "divide and share" approach works better than
/// trying to process all frequencies at once.
/// </para>
/// </remarks>
public class BandSplitRNNOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Gets or sets the number of frequency bins.</summary>
    public int NumFreqBins { get; set; } = 1025;

    #endregion

    #region Band Configuration

    /// <summary>Gets or sets the number of frequency bands to split into.</summary>
    /// <remarks>
    /// <para>
    /// The spectrogram is divided into this many non-overlapping frequency bands.
    /// Each band is processed independently by the band-level RNN before fusion.
    /// More bands allow finer frequency resolution but increase computation.
    /// </para>
    /// </remarks>
    public int NumBands { get; set; } = 24;

    /// <summary>Gets or sets the hidden size of the band-level RNN.</summary>
    public int BandRnnHiddenSize { get; set; } = 128;

    /// <summary>Gets or sets the number of band-level RNN layers.</summary>
    public int NumBandRnnLayers { get; set; } = 12;

    #endregion

    #region Sequence-Level Processing

    /// <summary>Gets or sets the hidden size of the sequence-level (cross-band) RNN.</summary>
    public int SequenceRnnHiddenSize { get; set; } = 256;

    /// <summary>Gets or sets the number of sequence-level RNN layers.</summary>
    public int NumSequenceRnnLayers { get; set; } = 6;

    /// <summary>Gets or sets the band fusion hidden dimension.</summary>
    public int FusionDim { get; set; } = 256;

    #endregion

    #region Separation

    /// <summary>Gets or sets the source names to separate.</summary>
    public string[] Sources { get; set; } = ["vocals", "drums", "bass", "other"];

    /// <summary>Gets or sets the number of stems/sources.</summary>
    public int NumStems { get; set; } = 4;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 1e-2;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
