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
    /// <summary>Creates options with the default values.</summary>
    public TESTOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public TESTOptions(TESTOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
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

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 512;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Number of time steps in each patch the series is split into.</summary>
    public int PatchLength { get; set; } = 16;
    /// <summary>Width of the model's hidden representations.</summary>
    public int HiddenDimension { get; set; } = 768;
    /// <summary>Number of transformer layers.</summary>
    public int NumLayers { get; set; } = 6;
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Preset size of the foundation model.</summary>
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
