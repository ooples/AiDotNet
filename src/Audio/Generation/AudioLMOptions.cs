using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the AudioLM audio generation model.
/// </summary>
/// <remarks>
/// <para>
/// AudioLM (Borsos et al., 2023, Google) generates high-quality, coherent audio by
/// combining semantic tokens (from a self-supervised model like w2v-BERT) with acoustic
/// tokens (from a neural codec like SoundStream). A hierarchical language model generates
/// semantic tokens first for high-level structure, then acoustic tokens for fine detail.
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioLM generates natural-sounding audio by "thinking" about
/// what to say/play at two levels: first the big-picture meaning (semantic), then the
/// fine details of how it sounds (acoustic). This two-stage approach produces audio that
/// is both coherent and high-fidelity, like a writer who first outlines a story then
/// adds the vivid details.
/// </para>
/// </remarks>
public class AudioLMOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the maximum generation duration in seconds.</summary>
    public double MaxDurationSeconds { get; set; } = 30.0;

    #endregion

    #region Semantic Stage

    /// <summary>Gets or sets the semantic token vocabulary size (from w2v-BERT).</summary>
    public int SemanticVocabSize { get; set; } = 1024;

    /// <summary>Gets or sets the semantic model dimension.</summary>
    public int SemanticDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of semantic transformer layers.</summary>
    public int NumSemanticLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of semantic attention heads.</summary>
    public int NumSemanticHeads { get; set; } = 16;

    /// <summary>Gets or sets the semantic token frame rate (tokens/second).</summary>
    public int SemanticFrameRate { get; set; } = 25;

    #endregion

    #region Coarse Acoustic Stage

    /// <summary>Gets or sets the coarse acoustic codebook size.</summary>
    public int CoarseCodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the number of coarse quantizers.</summary>
    public int NumCoarseQuantizers { get; set; } = 4;

    /// <summary>Gets or sets the coarse model dimension.</summary>
    public int CoarseDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of coarse transformer layers.</summary>
    public int NumCoarseLayers { get; set; } = 12;

    #endregion

    #region Fine Acoustic Stage

    /// <summary>Gets or sets the fine acoustic codebook size.</summary>
    public int FineCodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the total number of fine quantizers.</summary>
    public int NumFineQuantizers { get; set; } = 8;

    /// <summary>Gets or sets the fine model dimension.</summary>
    public int FineDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of fine transformer layers.</summary>
    public int NumFineLayers { get; set; } = 12;

    #endregion

    #region Generation

    /// <summary>Gets or sets the temperature for sampling.</summary>
    public double Temperature { get; set; } = 0.8;

    /// <summary>Gets or sets the top-k sampling parameter.</summary>
    public int TopK { get; set; } = 250;

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
