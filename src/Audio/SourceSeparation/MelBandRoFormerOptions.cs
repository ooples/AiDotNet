using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Configuration options for the MelBand-RoFormer model.
/// </summary>
/// <remarks>
/// <para>
/// MelBand-RoFormer (2024) extends BS-RoFormer by using mel-scale frequency bands instead of
/// linear bands, better matching human perception. It achieves state-of-the-art SDR on vocals
/// (13.2 dB) and other stems on the MUSDB18-HQ benchmark.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is an improved version of BS-RoFormer that divides frequencies
/// using the mel scale (which matches how humans hear) instead of equal-width bands.
/// This means more detail in the frequencies that matter most to us.
/// </para>
/// </remarks>
public class MelBandRoFormerOptions : ModelOptions
{
    #region Audio Preprocessing

    public int SampleRate { get; set; } = 44100;
    public int FftSize { get; set; } = 2048;
    public int HopLength { get; set; } = 512;
    public int NumFreqBins { get; set; } = 1025;
    public int NumMelBands { get; set; } = 60;

    #endregion

    #region Transformer Architecture

    public int NumTransformerLayers { get; set; } = 12;
    public int TransformerDim { get; set; } = 384;
    public int NumAttentionHeads { get; set; } = 8;
    public int FeedForwardDim { get; set; } = 1536;
    public int BandEmbeddingDim { get; set; } = 128;
    public bool UseRotaryEmbedding { get; set; } = true;
    public double DropoutRate { get; set; } = 0.0;

    #endregion

    #region Separation

    public string[] Sources { get; set; } = ["vocals", "drums", "bass", "other"];
    public int NumStems { get; set; } = 4;

    #endregion

    #region Model Loading

    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    public double LearningRate { get; set; } = 5e-5;
    public double WeightDecay { get; set; } = 1e-2;

    #endregion
}
