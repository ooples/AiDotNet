using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// Configuration options for the Audio Flamingo 2 model.
/// </summary>
/// <remarks>
/// <para>
/// Audio Flamingo 2 (2024) extends the Flamingo architecture for audio understanding with
/// interleaved audio-text inputs. It uses a frozen audio encoder with perceiver-style
/// cross-attention to adapt a pre-trained LLM for audio captioning, QA, and reasoning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio Flamingo 2 is like giving a language AI the ability to hear.
/// It can listen to audio recordings and answer questions about them, generate descriptions,
/// or reason about what's happening in the audio scene.
/// </para>
/// </remarks>
public class AudioFlamingo2Options : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the audio encoder dimension.</summary>
    public int AudioEncoderDim { get; set; } = 768;

    /// <summary>Gets or sets the LLM hidden dimension.</summary>
    public int LLMHiddenDim { get; set; } = 2048;

    /// <summary>Gets or sets the number of perceiver layers.</summary>
    public int NumPerceiverLayers { get; set; } = 2;

    /// <summary>Gets or sets the number of perceiver latent tokens.</summary>
    public int NumPerceiverTokens { get; set; } = 64;

    /// <summary>Gets or sets the maximum audio duration in seconds.</summary>
    public double MaxAudioDurationSeconds { get; set; } = 30.0;

    /// <summary>Gets or sets the maximum response tokens.</summary>
    public int MaxResponseTokens { get; set; } = 256;

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

    #endregion
}
