using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for genre classification.
/// </summary>
public class GenreClassifierOptions
{
    /// <summary>Audio sample rate. Default: 22050.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>FFT size. Default: 2048.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Hop length. Default: 512.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Number of MFCCs to extract. Default: 20.</summary>
    public int NumMfccs { get; set; } = 20;

    /// <summary>Number of top predictions to return. Default: 3.</summary>
    public int TopK { get; set; } = 3;

    /// <summary>Custom genre labels (optional).</summary>
    public string[]? CustomGenres { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
