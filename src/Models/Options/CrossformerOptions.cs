using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Crossformer model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Crossformer uses a cross-dimension attention mechanism that captures both temporal
/// and cross-variable dependencies simultaneously through a two-stage attention structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how the Crossformer model behaves:
///
/// Key settings:
/// - <b>SegmentLength:</b> How long each time segment is for cross-time attention.
/// - <b>NumLayers:</b> How deep the transformer is. More layers = more capacity but slower.
/// - <b>NumHeads:</b> Attention heads. More heads can capture different patterns.
/// - <b>RouterTopK:</b> For mixture of experts, how many experts to use per token.
///
/// Crossformer excels at capturing both temporal patterns and cross-variable relationships
/// that are common in multivariate financial time series.
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "Crossformer: Transformer Utilizing Cross-Dimension
/// Dependency for Multivariate Time Series Forecasting", ICLR 2023.
/// https://openreview.net/forum?id=vSVLM2j9eie
/// </para>
/// </remarks>
public class CrossformerOptions<T> : ModelOptions
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the segment length for cross-time attention.
    /// Default: 12.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The segment length determines how the time dimension is divided for
    /// hierarchical cross-time attention.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Crossformer divides time into segments to make attention
    /// more efficient. A segment length of 12 means every 12 time steps are grouped.
    /// </para>
    /// </remarks>
    public int SegmentLength { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// Default: 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More layers allow the model to learn more complex patterns but increase
    /// computation time and memory usage.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// Default: 4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multiple attention heads allow the model to attend to different parts of
    /// the sequence simultaneously, capturing various types of patterns.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the model dimension (embedding size).
    /// Default: 128.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model dimension determines the size of the internal representations.
    /// Larger dimensions can capture more complex patterns but require more memory.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the feedforward network dimension.
    /// Default: 256 (2x model dimension).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The FFN dimension in the transformer encoder layers.
    /// Typically 2-4 times the model dimension.
    /// </para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the top-K value for router in cross-dimension attention.
    /// Default: 2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Crossformer uses a routing mechanism similar to mixture
    /// of experts. TopK determines how many "experts" are used for each input.
    /// </para>
    /// </remarks>
    public int RouterTopK { get; set; } = 2;

    #endregion

    #region Regularization

    /// <summary>
    /// Gets or sets the dropout rate.
    /// Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout randomly zeros out a fraction of neurons during training,
    /// helping prevent overfitting.
    /// </para>
    /// </remarks>
    public double Dropout { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout applied to attention weights.
    /// </para>
    /// </remarks>
    public double AttentionDropout { get; set; } = 0.1;

    #endregion

    #region Normalization

    /// <summary>
    /// Gets or sets whether to use instance normalization (RevIN).
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instance normalization (RevIN) helps handle distribution shift in time series data
    /// by normalizing each instance to zero mean and unit variance.
    /// </para>
    /// </remarks>
    public bool UseInstanceNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pre-norm (LayerNorm before attention/FFN).
    /// Default: true.
    /// </summary>
    public bool UsePreNorm { get; set; } = true;

    #endregion

    #region Training Configuration

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses MSE loss).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the learning rate.
    /// Default: 0.0001.
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// Default: null (random initialization).
    /// </summary>
    public int? RandomSeed { get; set; }

    #endregion

    #region Input/Output Configuration

    /// <summary>
    /// Gets or sets the input sequence length.
    /// Default: 96.
    /// </summary>
    public int SequenceLength { get; set; } = 96;

    /// <summary>
    /// Gets or sets the prediction horizon.
    /// Default: 24.
    /// </summary>
    public int PredictionHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of input features.
    /// Default: 7.
    /// </summary>
    public int NumFeatures { get; set; } = 7;

    #endregion

    #region Validation

    /// <summary>
    /// Validates the options and returns any validation errors.
    /// </summary>
    /// <returns>List of validation error messages, empty if valid.</returns>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (SegmentLength < 1)
            errors.Add("SegmentLength must be at least 1.");
        if (NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
        if (NumHeads < 1)
            errors.Add("NumHeads must be at least 1.");
        if (ModelDimension < 1)
            errors.Add("ModelDimension must be at least 1.");
        if (ModelDimension % NumHeads != 0)
            errors.Add("ModelDimension must be divisible by NumHeads.");
        if (FeedForwardDimension < 1)
            errors.Add("FeedForwardDimension must be at least 1.");
        if (RouterTopK < 1)
            errors.Add("RouterTopK must be at least 1.");
        if (Dropout < 0.0 || Dropout >= 1.0)
            errors.Add("Dropout must be in [0, 1).");
        if (AttentionDropout < 0.0 || AttentionDropout >= 1.0)
            errors.Add("AttentionDropout must be in [0, 1).");
        if (LearningRate <= 0)
            errors.Add("LearningRate must be positive.");
        if (SequenceLength < SegmentLength)
            errors.Add("SequenceLength must be at least SegmentLength.");
        if (PredictionHorizon < 1)
            errors.Add("PredictionHorizon must be at least 1.");
        if (NumFeatures < 1)
            errors.Add("NumFeatures must be at least 1.");

        return errors;
    }

    /// <summary>
    /// Calculates the number of segments based on sequence length and segment length.
    /// </summary>
    /// <returns>The number of segments.</returns>
    public int CalculateNumSegments()
    {
        return (SequenceLength + SegmentLength - 1) / SegmentLength;
    }

    #endregion
}
