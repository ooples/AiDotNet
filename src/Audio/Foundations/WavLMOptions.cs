using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Configuration options for the WavLM self-supervised speech model.
/// </summary>
/// <remarks>
/// <para>
/// WavLM (Chen et al., 2022, Microsoft) extends HuBERT with gated relative position bias
/// and denoising pre-training. It achieves state-of-the-art on the SUPERB benchmark across
/// speech recognition, speaker verification, speaker diarization, and more.
/// </para>
/// <para>
/// <b>For Beginners:</b> WavLM is an improved version of HuBERT that's especially good at
/// understanding noisy speech and telling different speakers apart. It was trained to
/// understand speech even with background noise, making it more robust in real-world
/// conditions.
/// </para>
/// </remarks>
public class WavLMOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("base", "base+", or "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the transformer hidden dimension.</summary>
    public int HiddenDim { get; set; } = 768;

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Gets or sets the CNN feature encoder output dimension.</summary>
    public int FeatureEncoderDim { get; set; } = 512;

    /// <summary>Gets or sets whether to use gated relative position bias.</summary>
    public bool UseGatedRelativePositionBias { get; set; } = true;

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

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
