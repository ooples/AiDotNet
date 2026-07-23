using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Configuration options for the MERT music understanding foundation model.
/// </summary>
/// <remarks>
/// <para>
/// MERT (Li et al., 2024) is a self-supervised music understanding model that uses acoustic
/// and musical tokenizers to learn rich music representations. Unlike speech-focused models,
/// MERT incorporates music-specific knowledge through CQT-based teacher targets and codebook
/// clustering, enabling strong performance on 14 music information retrieval tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> MERT is like HuBERT but specialized for music instead of speech.
/// While HuBERT learns by predicting speech units, MERT learns by predicting musical features
/// like pitch and harmony. This means it deeply understands music structure and can be used
/// for tasks like genre classification, instrument detection, and music tagging.
/// </para>
/// </remarks>
public class MERTOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model hidden dimension.</summary>
    public int HiddenDim { get; set; } = 768;

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the feed-forward inner dimension.</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Gets or sets the model variant ("base" with 95M params or "large" with 330M).</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the convolutional feature extractor channels.</summary>
    public int[] ConvChannels { get; set; } = [512, 512, 512, 512, 512, 512, 512];

    #endregion

    #region Music-Specific

    /// <summary>Gets or sets the number of CQT bins for music teacher targets.</summary>
    public int CQTBins { get; set; } = 336;

    /// <summary>Gets or sets the number of RVQ codebooks for acoustic teacher.</summary>
    public int NumCodebooks { get; set; } = 8;

    /// <summary>Gets or sets the codebook vocabulary size.</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the number of K-means clusters for target quantization.</summary>
    public int NumClusters { get; set; } = 500;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the masking probability.</summary>
    public double MaskProbability { get; set; } = 0.8;

    /// <summary>Gets or sets the mask span length.</summary>
    public int MaskSpanLength { get; set; } = 10;

    #endregion
}
