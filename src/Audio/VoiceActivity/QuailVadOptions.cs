using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Configuration options for the Quail VAD model.
/// </summary>
/// <remarks>
/// <para>
/// Quail VAD (2024) is a lightweight voice activity detection model optimized for
/// on-device deployment. It uses a compact CNN-RNN architecture with knowledge distillation
/// from larger models, achieving high accuracy with minimal computational overhead.
/// </para>
/// <para>
/// <b>For Beginners:</b> Quail VAD detects when someone is speaking in audio - like a smart
/// "is anyone talking right now?" detector. It's designed to be small and fast enough to run
/// on phones and embedded devices in real-time.
/// </para>
/// </remarks>
public class QuailVadOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the hidden dimension.</summary>
    public int HiddenDim { get; set; } = 64;

    /// <summary>Gets or sets the number of CNN layers.</summary>
    public int NumCNNLayers { get; set; } = 3;

    /// <summary>Gets or sets the RNN hidden size.</summary>
    public int RNNHiddenSize { get; set; } = 64;

    /// <summary>Gets or sets the frame size in milliseconds.</summary>
    public int FrameSizeMs { get; set; } = 30;

    #endregion

    #region Detection

    /// <summary>Gets or sets the speech detection threshold.</summary>
    public double Threshold { get; set; } = 0.5;

    /// <summary>Gets or sets the minimum speech duration in seconds.</summary>
    public double MinSpeechDuration { get; set; } = 0.25;

    /// <summary>Gets or sets the minimum silence duration in seconds.</summary>
    public double MinSilenceDuration { get; set; } = 0.1;

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
