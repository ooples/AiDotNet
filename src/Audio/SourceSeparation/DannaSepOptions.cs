using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Configuration options for the Danna-Sep (Dual-path Attention Neural Network Audio Separator) model.
/// </summary>
/// <remarks>
/// <para>
/// Danna-Sep (2024) uses dual-path attention with interleaved intra-chunk and inter-chunk
/// processing for music source separation. It achieves competitive results on MUSDB18 by
/// efficiently modeling both local spectral patterns and long-range temporal dependencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> Danna-Sep separates mixed music into individual instruments by
/// looking at audio in two ways: short-range patterns (what notes are being played right now)
/// and long-range patterns (how the music evolves over time). This dual perspective helps it
/// accurately pull apart overlapping instruments.
/// </para>
/// </remarks>
public class DannaSepOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the number of audio channels.</summary>
    public int NumChannels { get; set; } = 2;

    /// <summary>Gets or sets the FFT size for STFT computation.</summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>Gets or sets the hop length for STFT computation.</summary>
    public int HopLength { get; set; } = 1024;

    /// <summary>Gets or sets the number of frequency bins (FftSize/2 + 1).</summary>
    public int NumFreqBins { get; set; } = 2049;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the encoder dimension.</summary>
    public int EncoderDim { get; set; } = 256;

    /// <summary>Gets or sets the number of dual-path blocks.</summary>
    public int NumDualPathBlocks { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the chunk size for dual-path processing.</summary>
    public int ChunkSize { get; set; } = 250;

    /// <summary>Gets or sets the number of sources to separate.</summary>
    public int NumSources { get; set; } = 4;

    /// <summary>Gets or sets the source names.</summary>
    public string[] SourceNames { get; set; } = ["vocals", "drums", "bass", "other"];

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
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
