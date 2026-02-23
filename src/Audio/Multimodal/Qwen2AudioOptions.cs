using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// Configuration options for the Qwen2-Audio multimodal audio-language model.
/// </summary>
/// <remarks>
/// <para>
/// Qwen2-Audio (Chu et al., 2024, Alibaba) is a large audio-language model that can process
/// audio and text inputs to generate text responses. It uses a Whisper-style audio encoder
/// with a Qwen2 language model backbone, connected by a perceiver-style adapter. It supports
/// audio captioning, question answering, sound event detection, and audio reasoning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Qwen2-Audio is like having a conversation partner who can hear.
/// You play it audio and ask questions like "What sounds do you hear?" or "Describe this music."
/// It uses a powerful language model (similar to ChatGPT) combined with an audio understanding
/// system to provide intelligent responses about any audio input.
/// </para>
/// </remarks>
public class Qwen2AudioOptions : ModelOptions
{
    #region Audio Encoder

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the audio encoder dimension (Whisper-style).</summary>
    public int AudioEncoderDim { get; set; } = 1280;

    /// <summary>Gets or sets the number of audio encoder layers.</summary>
    public int NumAudioEncoderLayers { get; set; } = 32;

    /// <summary>Gets or sets the number of audio encoder attention heads.</summary>
    public int NumAudioEncoderHeads { get; set; } = 20;

    /// <summary>Gets or sets the number of mel spectrogram channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the maximum audio duration in seconds.</summary>
    public double MaxAudioDurationSeconds { get; set; } = 30.0;

    #endregion

    #region Language Model

    /// <summary>Gets or sets the language model hidden dimension.</summary>
    public int LMHiddenDim { get; set; } = 3584;

    /// <summary>Gets or sets the number of language model layers.</summary>
    public int NumLMLayers { get; set; } = 28;

    /// <summary>Gets or sets the number of language model attention heads.</summary>
    public int NumLMHeads { get; set; } = 28;

    /// <summary>Gets or sets the LM vocabulary size.</summary>
    public int VocabSize { get; set; } = 151936;

    /// <summary>Gets or sets the maximum response length in tokens.</summary>
    public int MaxResponseTokens { get; set; } = 512;

    #endregion

    #region Adapter

    /// <summary>Gets or sets the perceiver adapter output dimension.</summary>
    public int AdapterDim { get; set; } = 3584;

    /// <summary>Gets or sets the number of perceiver latent tokens.</summary>
    public int NumLatentTokens { get; set; } = 64;

    #endregion

    #region Generation

    /// <summary>Gets or sets the sampling temperature.</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Gets or sets the top-p (nucleus) sampling parameter.</summary>
    public double TopP { get; set; } = 0.9;

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
