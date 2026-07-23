using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// Configuration options for the Wav2Small speech emotion recognition model.
/// </summary>
/// <remarks>
/// <para>
/// Wav2Small (Gomez-Alanis et al., 2024) is a lightweight speech emotion recognition model
/// that distills knowledge from large wav2vec 2.0 models into a compact architecture. It achieves
/// competitive accuracy on IEMOCAP and RAVDESS while being small enough for edge deployment.
/// </para>
/// <para>
/// <b>For Beginners:</b> Wav2Small is like a student who learned from a large, expert teacher.
/// The teacher (wav2vec 2.0) is too big to run on phones or embedded devices, so Wav2Small
/// learns the teacher's emotion detection skills in a much smaller model. The result is fast,
/// accurate emotion detection that can run on resource-limited devices.
/// </para>
/// </remarks>
public class Wav2SmallOptions : ModelOptions
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

    #region Architecture

    /// <summary>Gets or sets the hidden dimension of the compact encoder.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the feed-forward hidden dimension.</summary>
    public int FeedForwardDim { get; set; } = 1024;

    /// <summary>Gets or sets the CNN feature encoder output dimension.</summary>
    public int FeatureEncoderDim { get; set; } = 256;

    #endregion

    #region Emotion Classes

    /// <summary>Gets or sets the number of emotion classes.</summary>
    public int NumClasses { get; set; } = 7;

    /// <summary>Gets or sets the emotion label names.</summary>
    public string[] EmotionLabels { get; set; } = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];

    /// <summary>Gets or sets whether to include arousal/valence estimation.</summary>
    public bool IncludeArousalValence { get; set; } = true;

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

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the knowledge distillation temperature.</summary>
    public double DistillationTemperature { get; set; } = 2.0;

    #endregion
}
