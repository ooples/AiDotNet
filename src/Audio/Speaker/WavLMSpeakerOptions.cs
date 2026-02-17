using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the WavLM Speaker verification and embedding model.
/// </summary>
/// <remarks>
/// <para>
/// WavLM (Chen et al., 2022) is a self-supervised speech model that, when fine-tuned for speaker
/// verification, achieves state-of-the-art results with 0.59% EER on VoxCeleb1 test set. It uses
/// a Transformer encoder pre-trained with masked speech prediction and denoising objectives,
/// making it robust to noisy conditions.
/// </para>
/// <para>
/// <b>For Beginners:</b> WavLM was originally trained to understand speech in general (like a
/// language student who listens to lots of conversations). When specialized for speaker verification,
/// it becomes excellent at recognizing individual voices because it already understands the deep
/// structure of speech. It works especially well in noisy environments.
/// </para>
/// </remarks>
public class WavLMSpeakerOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 160;

    #endregion

    #region WavLM Architecture

    /// <summary>Gets or sets the model variant ("base" or "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the Transformer hidden dimension.</summary>
    public int HiddenDim { get; set; } = 768;

    /// <summary>Gets or sets the number of Transformer layers.</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Gets or sets the feed-forward hidden dimension.</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Gets or sets the feature encoder dimension (CNN output).</summary>
    public int FeatureEncoderDim { get; set; } = 512;

    #endregion

    #region Embedding

    /// <summary>Gets or sets the output speaker embedding dimension.</summary>
    public int EmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the pooling strategy ("mean", "stats", or "attentive").</summary>
    public string PoolingStrategy { get; set; } = "stats";

    #endregion

    #region Verification

    /// <summary>Gets or sets the default cosine similarity threshold.</summary>
    public double DefaultThreshold { get; set; } = 0.65;

    /// <summary>Gets or sets the minimum audio duration in seconds.</summary>
    public double MinDurationSeconds { get; set; } = 0.5;

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

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 1e-2;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
