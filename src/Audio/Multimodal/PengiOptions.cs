using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// Configuration options for the Pengi model.
/// </summary>
/// <remarks>
/// <para>
/// Pengi (Deshmukh et al., 2023, Microsoft) is an audio language model that frames all audio
/// tasks as text-generation tasks. It uses a frozen audio encoder (Audio Spectrogram Transformer)
/// paired with a pre-trained language model, enabling open-ended audio reasoning, captioning,
/// and question answering without task-specific heads.
/// </para>
/// <para>
/// <b>For Beginners:</b> Pengi treats all audio understanding as a conversation. Instead of
/// having separate models for "what sound is this?" and "describe this audio", Pengi uses one
/// model that can answer any question about audio by generating text responses.
/// </para>
/// </remarks>
public class PengiOptions : ModelOptions
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

    /// <summary>Gets or sets the number of projection layers from audio to LLM space.</summary>
    public int NumProjectionLayers { get; set; } = 2;

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
