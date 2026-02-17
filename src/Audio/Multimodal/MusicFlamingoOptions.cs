using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// Configuration options for the Music Flamingo model.
/// </summary>
/// <remarks>
/// <para>
/// Music Flamingo (2024) adapts the Flamingo architecture specifically for music understanding.
/// It uses a frozen music encoder (e.g., MERT or Jukebox features) with perceiver cross-attention
/// to enable a pre-trained LLM to reason about music: answering questions about genre, instruments,
/// mood, structure, and musical theory.
/// </para>
/// <para>
/// <b>For Beginners:</b> Music Flamingo is like giving a language AI the ability to understand music.
/// You can play it a song and ask "What genre is this?" or "What instruments are playing?" and it
/// will answer in natural language, combining its music listening ability with language understanding.
/// </para>
/// </remarks>
public class MusicFlamingoOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the music encoder dimension.</summary>
    public int MusicEncoderDim { get; set; } = 768;

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
