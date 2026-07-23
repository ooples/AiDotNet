using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the SpeakerLM model.
/// </summary>
/// <remarks>
/// <para>
/// SpeakerLM (2024) uses a language model backbone for speaker understanding tasks including
/// speaker verification, diarization, and speaker-attributed transcription. It processes
/// speaker embeddings as tokens in a sequence, enabling multi-speaker understanding in a
/// unified framework.
/// </para>
/// <para>
/// <b>For Beginners:</b> SpeakerLM treats speaker recognition like a language problem - it
/// "reads" speaker characteristics the way a language model reads words. This lets it handle
/// complex multi-speaker scenarios where multiple people are talking.
/// </para>
/// </remarks>
public class SpeakerLMOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel-frequency bins for audio preprocessing.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the minimum audio duration in seconds for reliable embedding extraction.</summary>
    public double MinDurationSeconds { get; set; } = 0.5;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the speaker embedding dimension.</summary>
    public int EmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the language model hidden dimension.</summary>
    public int LMHiddenDim { get; set; } = 768;

    /// <summary>Gets or sets the number of LM layers.</summary>
    public int NumLMLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the maximum number of speakers.</summary>
    public int MaxSpeakers { get; set; } = 16;

    /// <summary>Gets or sets the default verification threshold (cosine similarity).</summary>
    public double DefaultThreshold { get; set; } = 0.7;

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
