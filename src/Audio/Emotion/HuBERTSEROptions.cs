using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// Configuration options for the HuBERT-based Speech Emotion Recognition model.
/// </summary>
/// <remarks>
/// <para>
/// HuBERT-SER fine-tunes the HuBERT (Hsu et al., 2021) self-supervised model for speech
/// emotion recognition. HuBERT learns speech representations through masked prediction,
/// and when fine-tuned for SER, achieves strong results on IEMOCAP (69.7% weighted accuracy).
/// </para>
/// <para>
/// <b>For Beginners:</b> HuBERT-SER uses a pre-trained speech understanding model (HuBERT)
/// and teaches it to recognize emotions. HuBERT first learns general speech patterns from
/// millions of hours of audio, then is specialized for emotion detection.
/// </para>
/// </remarks>
public class HuBERTSEROptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    public int NumMels { get; set; } = 80;

    /// <summary>
    /// Gets or sets the FFT window size in samples.
    /// </summary>
    public int FftSize { get; set; } = 400;

    /// <summary>
    /// Gets or sets the hop length between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 160;

    #endregion

    #region HuBERT Architecture

    /// <summary>
    /// Gets or sets the model variant (base or large).
    /// </summary>
    public string Variant { get; set; } = "base";

    /// <summary>
    /// Gets or sets the Transformer encoder dimension.
    /// </summary>
    public int TransformerDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of Transformer layers.
    /// </summary>
    public int NumTransformerLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the feed-forward dimension.
    /// </summary>
    public int FeedForwardDim { get; set; } = 3072;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the emotion labels.
    /// </summary>
    public string[] EmotionLabels { get; set; } = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];

    /// <summary>
    /// Gets or sets the number of emotion classes.
    /// </summary>
    public int NumClasses { get; set; } = 7;

    /// <summary>
    /// Gets or sets the classification head hidden dimension.
    /// </summary>
    public int ClassifierHiddenDim { get; set; } = 256;

    /// <summary>
    /// Gets or sets whether to include arousal/valence regression heads.
    /// </summary>
    public bool IncludeArousalValence { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
