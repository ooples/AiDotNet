using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Configuration options for the WebRTC VAD neural model.
/// </summary>
/// <remarks>
/// <para>
/// WebRTC VAD is a lightweight voice activity detection model inspired by the GMM-based
/// detector in the WebRTC framework but reimplemented as a neural network for improved accuracy.
/// It operates at very low latency (10-30ms frames) and is designed for real-time communication.
/// </para>
/// <para>
/// <b>For Beginners:</b> WebRTC VAD is a very fast "is someone talking?" detector used in
/// video calls and voice chat. It processes tiny chunks of audio (10-30 milliseconds) and
/// instantly decides if speech is present. Speed matters most here - it needs to work in
/// real-time without adding noticeable delay to your call.
/// </para>
/// </remarks>
public class WebRTCVadOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the frame duration in milliseconds (10, 20, or 30).</summary>
    public int FrameDurationMs { get; set; } = 30;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the hidden dimension.</summary>
    public int HiddenDim { get; set; } = 64;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumLayers { get; set; } = 3;

    /// <summary>Gets or sets the aggressiveness mode (0-3, higher = more aggressive filtering).</summary>
    public int AggressivenessMode { get; set; } = 1;

    /// <summary>Gets or sets the detection threshold.</summary>
    public double Threshold { get; set; } = 0.5;

    /// <summary>Gets or sets the minimum speech duration in milliseconds.</summary>
    public int MinSpeechDurationMs { get; set; } = 250;

    /// <summary>Gets or sets the minimum silence duration in milliseconds.</summary>
    public int MinSilenceDurationMs { get; set; } = 100;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

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

    #endregion
}
