using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the ETSformer model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// ETSformer (Exponential Smoothing Transformer) combines classical exponential smoothing
/// methods with transformer attention mechanisms for interpretable time series forecasting.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how the ETSformer model behaves:
///
/// Key settings:
/// - <b>NumEncoderLayers:</b> How many encoding layers to use for pattern extraction
/// - <b>NumDecoderLayers:</b> How many decoding layers for generating forecasts
/// - <b>K:</b> Top-K frequencies for seasonal pattern detection
/// - <b>LevelSmoothing:</b> Controls how quickly the model adapts to level changes
///
/// ETSformer is particularly interpretable because it explicitly models trend, seasonality,
/// and growth components that you can inspect and understand.
/// </para>
/// <para>
/// <b>Reference:</b> Woo et al., "ETSformer: Exponential Smoothing Transformers for
/// Time-series Forecasting", 2022. https://arxiv.org/abs/2202.01381
/// </para>
/// </remarks>
public class ETSformerOptions<T>
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
    /// ETSformer typically uses fewer decoder layers than encoder layers.
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

    /// <summary>
    /// Gets or sets the top-K frequencies for seasonal decomposition.
    /// Default: 5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> K determines how many dominant seasonal frequencies
    /// to extract. Higher values capture more complex seasonality patterns.
    /// </para>
    /// </remarks>
    public int K { get; set; } = 5;

    #endregion

    #region Exponential Smoothing Parameters

    /// <summary>
    /// Gets or sets the level smoothing factor (alpha).
    /// Default: 0.5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how quickly the level component adapts to changes.
    /// Higher values (closer to 1.0) make the model more responsive to recent changes.
    /// </para>
    /// </remarks>
    public double LevelSmoothing { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the trend smoothing factor (beta).
    /// Default: 0.5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how quickly the trend component adapts.
    /// Higher values make trend estimation more responsive to changes.
    /// </para>
    /// </remarks>
    public double TrendSmoothing { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the seasonal smoothing factor (gamma).
    /// Default: 0.5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how quickly seasonal patterns adapt.
    /// Higher values allow seasonal patterns to change more rapidly.
    /// </para>
    /// </remarks>
    public double SeasonalSmoothing { get; set; } = 0.5;

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
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instance normalization (RevIN) helps handle distribution shift in time series data.
    /// </para>
    /// </remarks>
    public bool UseInstanceNormalization { get; set; } = true;

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
        if (K < 1)
            errors.Add("K must be at least 1.");
        if (LevelSmoothing < 0.0 || LevelSmoothing > 1.0)
            errors.Add("LevelSmoothing must be in [0, 1].");
        if (TrendSmoothing < 0.0 || TrendSmoothing > 1.0)
            errors.Add("TrendSmoothing must be in [0, 1].");
        if (SeasonalSmoothing < 0.0 || SeasonalSmoothing > 1.0)
            errors.Add("SeasonalSmoothing must be in [0, 1].");
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

        return errors;
    }

    #endregion
}
