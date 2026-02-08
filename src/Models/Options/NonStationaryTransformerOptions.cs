using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Non-stationary Transformer model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Non-stationary Transformer addresses the over-stationarization problem in time series
/// forecasting by proposing Series Stationarization and De-stationary Attention mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how the Non-stationary Transformer model behaves:
///
/// Key settings:
/// - <b>NumEncoderLayers:</b> How many encoding layers to use for pattern extraction
/// - <b>NumDecoderLayers:</b> How many decoding layers for generating forecasts
/// - <b>NumHeads:</b> Number of attention heads for multi-head attention
/// - <b>UseDeStat:</b> Whether to use De-stationary Attention mechanism
///
/// Time series data often has changing statistical properties (non-stationarity). This model
/// explicitly handles this by:
/// 1. Normalizing the data to be stationary for better attention
/// 2. De-normalizing attention outputs to preserve original data characteristics
/// </para>
/// <para>
/// <b>Reference:</b> Liu et al., "Non-stationary Transformers: Exploring the Stationarity
/// in Time Series Forecasting", NeurIPS 2022. https://arxiv.org/abs/2205.14415
/// </para>
/// </remarks>
public class NonStationaryTransformerOptions<T> : ModelOptions
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// Default: 2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Encoder layers extract patterns from the input sequence.
    /// More layers can capture more complex patterns but increase computation.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// Default: 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decoder layers generate the forecast output.
    /// Typically uses fewer decoder layers than encoder layers.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the model dimension (embedding size).
    /// Default: 64.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model dimension determines the size of the internal representations.
    /// Larger dimensions can capture more complex patterns but require more memory.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the feedforward network dimension.
    /// Default: 128 (2x model dimension).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The FFN dimension in the transformer layers.
    /// Typically 2-4 times the model dimension.
    /// </para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// Default: 4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple attention heads allow the model to focus on
    /// different aspects of the input simultaneously, capturing various patterns.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    #endregion

    #region Non-stationary Mechanisms

    /// <summary>
    /// Gets or sets whether to use Series Stationarization.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Series Stationarization normalizes the input time series
    /// by removing its mean and scaling, making it easier for the model to learn patterns.
    /// </para>
    /// </remarks>
    public bool UseSeriesStationarization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use De-stationary Attention.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> De-stationary Attention rescales the attention weights
    /// using learned statistics to preserve original data characteristics while still
    /// benefiting from stationarized attention computation.
    /// </para>
    /// </remarks>
    public bool UseDeStationaryAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of projection dimensions for stationarization.
    /// Default: 64.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls the dimensionality of the learnable
    /// projections used in the de-stationarization process.
    /// </para>
    /// </remarks>
    public int ProjectionDimension { get; set; } = 64;

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

    #endregion

    #region Normalization

    /// <summary>
    /// Gets or sets whether to use instance normalization (RevIN).
    /// Default: false (uses Series Stationarization instead).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When false, uses the paper's Series Stationarization instead of RevIN.
    /// Set to true to use RevIN for comparison.
    /// </para>
    /// </remarks>
    public bool UseInstanceNormalization { get; set; } = false;

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

    /// <summary>
    /// Gets or sets the label length (decoder input overlap).
    /// Default: 48.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label length determines how much of the historical
    /// sequence is fed to the decoder as context. This helps the decoder
    /// understand the recent trend before making predictions.
    /// </para>
    /// </remarks>
    public int LabelLength { get; set; } = 48;

    #endregion

    #region Validation

    /// <summary>
    /// Validates the options and returns any validation errors.
    /// </summary>
    /// <returns>List of validation error messages, empty if valid.</returns>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (NumEncoderLayers < 1)
            errors.Add("NumEncoderLayers must be at least 1.");
        if (NumDecoderLayers < 1)
            errors.Add("NumDecoderLayers must be at least 1.");
        if (NumHeads < 1)
            errors.Add("NumHeads must be at least 1.");
        if (ModelDimension < 1)
            errors.Add("ModelDimension must be at least 1.");
        if (ModelDimension % NumHeads != 0)
            errors.Add("ModelDimension must be divisible by NumHeads.");
        if (FeedForwardDimension < 1)
            errors.Add("FeedForwardDimension must be at least 1.");
        if (ProjectionDimension < 1)
            errors.Add("ProjectionDimension must be at least 1.");
        if (Dropout < 0.0 || Dropout >= 1.0)
            errors.Add("Dropout must be in [0, 1).");
        if (LearningRate <= 0)
            errors.Add("LearningRate must be positive.");
        if (SequenceLength < 1)
            errors.Add("SequenceLength must be at least 1.");
        if (PredictionHorizon < 1)
            errors.Add("PredictionHorizon must be at least 1.");
        if (NumFeatures < 1)
            errors.Add("NumFeatures must be at least 1.");
        if (LabelLength < 0)
            errors.Add("LabelLength must be non-negative.");
        if (LabelLength > SequenceLength)
            errors.Add("LabelLength cannot exceed SequenceLength.");

        return errors;
    }

    #endregion
}
