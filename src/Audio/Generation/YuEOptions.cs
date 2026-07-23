using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the YuE music generation model.
/// </summary>
/// <remarks>
/// <para>
/// YuE (Yuan et al., 2025) is a full-song music generation model that generates complete songs
/// with vocals and accompaniment from lyrics and genre/style tags. It uses a dual-AR architecture
/// where a lyrics-conditioned language model generates semantic tokens and a second stage
/// produces acoustic tokens, generating songs of several minutes in length.
/// </para>
/// <para>
/// <b>For Beginners:</b> YuE is like having a virtual band that can write and perform an
/// entire song. You give it lyrics and a style ("pop, female vocalist, upbeat") and it generates
/// a complete song with singing, instruments, and production. Unlike most AI music tools that
/// only make short clips, YuE can create full-length songs.
/// </para>
/// </remarks>
public class YuEOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the maximum generation duration in seconds.</summary>
    public double MaxDurationSeconds { get; set; } = 300.0;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the semantic stage model dimension.</summary>
    public int SemanticDim { get; set; } = 2048;

    /// <summary>Gets or sets the number of semantic stage transformer layers.</summary>
    public int NumSemanticLayers { get; set; } = 24;

    /// <summary>Gets or sets the number of semantic stage attention heads.</summary>
    public int NumSemanticHeads { get; set; } = 16;

    /// <summary>Gets or sets the acoustic stage model dimension.</summary>
    public int AcousticDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of acoustic stage transformer layers.</summary>
    public int NumAcousticLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of acoustic stage attention heads.</summary>
    public int NumAcousticHeads { get; set; } = 16;

    /// <summary>Gets or sets the lyrics token vocabulary size.</summary>
    public int LyricsVocabSize { get; set; } = 32000;

    /// <summary>Gets or sets the semantic token vocabulary size.</summary>
    public int SemanticVocabSize { get; set; } = 16384;

    /// <summary>Gets or sets the acoustic codec codebook size.</summary>
    public int AcousticCodebookSize { get; set; } = 2048;

    /// <summary>Gets or sets the number of acoustic codec quantizers.</summary>
    public int NumAcousticQuantizers { get; set; } = 8;

    #endregion

    #region Genre and Style

    /// <summary>Gets or sets the number of genre/style tag embeddings.</summary>
    public int NumStyleTags { get; set; } = 200;

    /// <summary>Gets or sets the style embedding dimension.</summary>
    public int StyleEmbeddingDim { get; set; } = 512;

    #endregion

    #region Generation

    /// <summary>Gets or sets the temperature for sampling.</summary>
    public double Temperature { get; set; } = 0.9;

    /// <summary>Gets or sets the top-p (nucleus) sampling parameter.</summary>
    public double TopP { get; set; } = 0.95;

    /// <summary>Gets or sets the repetition penalty factor.</summary>
    public double RepetitionPenalty { get; set; } = 1.2;

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
