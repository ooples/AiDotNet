using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// Configuration options for the WavLM-SER speech emotion recognition model.
/// </summary>
/// <remarks>
/// <para>
/// WavLM-SER fine-tunes the WavLM self-supervised model (Chen et al., 2022) for speech emotion
/// recognition. WavLM's pre-training on masked speech prediction and denoising produces robust
/// features that, when fine-tuned for SER, achieve state-of-the-art results on IEMOCAP
/// (weighted accuracy 71%+) and are robust to noise and recording conditions.
/// </para>
/// <para>
/// <b>For Beginners:</b> WavLM-SER takes a model that first learned to understand speech in
/// general, then teaches it to recognize emotions specifically. Because WavLM already understands
/// the deep patterns of human speech, it picks up on subtle emotional cues that simpler models
/// miss—like the slight tremor in a fearful voice or the rise in pitch when someone is excited.
/// </para>
/// </remarks>
public class WavLMSEROptions : ModelOptions
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

    /// <summary>Gets or sets the CNN feature encoder output dimension.</summary>
    public int FeatureEncoderDim { get; set; } = 512;

    #endregion

    #region Emotion Classes

    /// <summary>Gets or sets the number of emotion classes.</summary>
    public int NumClasses { get; set; } = 7;

    /// <summary>Gets or sets the emotion label names.</summary>
    public string[] EmotionLabels { get; set; } = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];

    /// <summary>Gets or sets whether to include arousal/valence estimation.</summary>
    public bool IncludeArousalValence { get; set; } = true;

    /// <summary>Gets or sets which WavLM layer to use for features (-1 = weighted sum).</summary>
    public int FeatureLayerIndex { get; set; } = -1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
