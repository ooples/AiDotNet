using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TEST (Text Embedding for Seasonality and Trend — Generating Text-Aligned Embeddings for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TEST generates text-prototype-aligned embeddings for time series by leveraging pretrained
/// language model knowledge. It translates seasonal/trend patterns into text descriptions
/// and aligns time series embeddings to these text prototypes.
/// </para>
/// <para>
/// <b>Reference:</b> Sun et al., "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series", 2024.
/// </para>
/// </remarks>
public class TESTOptions<T> : TimeSeriesRegressionOptions<T>
{
    public TESTOptions() { }

    public TESTOptions(TESTOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        TextEmbeddingDimension = other.TextEmbeddingDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumPrototypes = other.NumPrototypes;
        AlignmentWeight = other.AlignmentWeight;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 16;
    public int HiddenDimension { get; set; } = 768;
    public int NumLayers { get; set; } = 6;
    public int NumHeads { get; set; } = 8;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the text embedding dimension from the language model.
    /// </summary>
    /// <value>Defaults to 768 (BERT-base dimension).</value>
    public int TextEmbeddingDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of text prototypes for alignment.
    /// </summary>
    /// <value>Defaults to 64.</value>
    public int NumPrototypes { get; set; } = 64;

    /// <summary>
    /// Gets or sets the weight for text-alignment contrastive loss.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double AlignmentWeight { get; set; } = 0.1;
}
