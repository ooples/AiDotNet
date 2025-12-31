using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for acoustic scene classification.
/// </summary>
public class SceneClassifierOptions
{
    /// <summary>Audio sample rate. Default: 22050.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>FFT size. Default: 2048.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Hop length. Default: 512.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Number of mel bands. Default: 128.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Number of MFCCs. Default: 20.</summary>
    public int NumMfccs { get; set; } = 20;

    /// <summary>Number of top predictions to return. Default: 3.</summary>
    public int TopK { get; set; } = 3;

    /// <summary>Custom scene labels (optional).</summary>
    public string[]? CustomScenes { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
