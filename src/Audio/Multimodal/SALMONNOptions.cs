using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// Configuration options for the SALMONN multimodal audio-language model.
/// </summary>
/// <remarks>
/// <para>
/// SALMONN (Tang et al., 2024, Tsinghua/ByteDance) is a large language model with dual
/// audio encoders: a Whisper speech encoder and a BEATs audio encoder, connected to a
/// Vicuna LLM through a window-level Q-Former adapter. This dual-encoder design gives it
/// strong capability for both speech understanding and general audio understanding tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> SALMONN has two "ears": one specialized for speech (Whisper) and
/// one for general sounds (BEATs). This means it can understand both what people say AND
/// non-speech sounds like music, animal sounds, and environmental noise. It's like having
/// a translator who also happens to be an expert sound engineer.
/// </para>
/// </remarks>
public class SALMONNOptions : ModelOptions
{
    #region Audio Encoders

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the speech encoder dimension (Whisper).</summary>
    public int SpeechEncoderDim { get; set; } = 1280;

    /// <summary>Gets or sets the number of speech encoder layers.</summary>
    public int NumSpeechEncoderLayers { get; set; } = 32;

    /// <summary>Gets or sets the audio encoder dimension (BEATs).</summary>
    public int AudioEncoderDim { get; set; } = 768;

    /// <summary>Gets or sets the number of audio encoder layers.</summary>
    public int NumAudioEncoderLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of mel spectrogram channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the maximum audio duration in seconds.</summary>
    public double MaxAudioDurationSeconds { get; set; } = 30.0;

    #endregion

    #region Q-Former Adapter

    /// <summary>Gets or sets the Q-Former hidden dimension.</summary>
    public int QFormerDim { get; set; } = 768;

    /// <summary>Gets or sets the number of Q-Former layers.</summary>
    public int NumQFormerLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of Q-Former query tokens.</summary>
    public int NumQueryTokens { get; set; } = 32;

    /// <summary>Gets or sets the window size for window-level Q-Former.</summary>
    public int WindowSize { get; set; } = 4;

    #endregion

    #region Language Model

    /// <summary>Gets or sets the language model hidden dimension.</summary>
    public int LMHiddenDim { get; set; } = 4096;

    /// <summary>Gets or sets the number of language model layers.</summary>
    public int NumLMLayers { get; set; } = 32;

    /// <summary>Gets or sets the number of language model attention heads.</summary>
    public int NumLMHeads { get; set; } = 32;

    /// <summary>Gets or sets the LM vocabulary size.</summary>
    public int VocabSize { get; set; } = 32000;

    /// <summary>Gets or sets the maximum response length in tokens.</summary>
    public int MaxResponseTokens { get; set; } = 512;

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
