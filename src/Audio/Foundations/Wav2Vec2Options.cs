using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Configuration options for the wav2vec 2.0 self-supervised speech model.
/// </summary>
/// <remarks>
/// <para>
/// wav2vec 2.0 (Baevski et al., 2020, Meta) learns speech representations via contrastive
/// learning over quantized speech units. Pre-trained on 960 hours of LibriSpeech, it achieves
/// WER 1.8% on test-clean with only 10 minutes of labeled data when fine-tuned for ASR.
/// </para>
/// <para>
/// <b>For Beginners:</b> wav2vec 2.0 pioneered self-supervised learning for speech. It
/// listens to raw audio, masks parts of it, and learns to predict the missing parts. This
/// teaches it a deep understanding of speech that can then be used for tasks like
/// transcription with very little labeled data.
/// </para>
/// </remarks>
public class Wav2Vec2Options : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("base" or "large").</summary>
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

    /// <summary>Gets or sets the number of quantization codebooks for contrastive learning.</summary>
    public int NumQuantizationGroups { get; set; } = 2;

    /// <summary>Gets or sets the quantization codebook size.</summary>
    public int QuantizationCodebookSize { get; set; } = 320;

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

    /// <summary>Gets or sets the contrastive loss temperature.</summary>
    public double ContrastiveTemperature { get; set; } = 0.1;

    #endregion
}
