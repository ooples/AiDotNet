using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TSMixer model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TSMixer is an all-MLP architecture for multivariate time series forecasting that
/// achieves state-of-the-art results using only multilayer perceptrons (MLPs) without
/// attention mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how the TSMixer model behaves:
///
/// Key settings:
/// - <b>NumBlocks:</b> Number of mixer blocks (similar to layers in other models)
/// - <b>HiddenDimension:</b> Size of hidden layers in the MLPs
/// - <b>FeaturesFirst:</b> Whether to mix features before time (affects performance)
/// - <b>UseRevIN:</b> Whether to use reversible instance normalization
///
/// TSMixer is simpler than transformer-based models but can be just as effective.
/// It's faster to train and uses less memory than attention-based approaches.
/// </para>
/// <para>
/// <b>Reference:</b> Chen et al., "TSMixer: An All-MLP Architecture for Time Series
/// Forecasting", TMLR 2023. https://arxiv.org/abs/2303.06053
/// </para>
/// </remarks>
public class TSMixerOptions<T> : ModelOptions
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of mixer blocks.
    /// Default: 4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each mixer block mixes information across time and features.
    /// More blocks can capture more complex patterns but increase computation.
    /// </para>
    /// </remarks>
    public int NumBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the hidden dimension for MLP layers.
    /// Default: 64.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The hidden dimension determines the size of intermediate
    /// representations in the MLP layers. Larger values can capture more complex patterns.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the feedforward expansion factor.
    /// Default: 2.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The feedforward layers expand the dimension temporarily
    /// by this factor, allowing the model to learn richer representations.
    /// </para>
    /// </remarks>
    public double FeedForwardExpansion { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets whether to process features before time dimension.
    /// Default: false (time-mixing first).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The order of mixing can affect results. Time-mixing first
    /// (false) focuses on temporal patterns, while feature-mixing first (true) focuses
    /// on cross-variable relationships.
    /// </para>
    /// </remarks>
    public bool FeaturesFirst { get; set; } = false;

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
    /// Gets or sets whether to use reversible instance normalization (RevIN).
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RevIN normalizes each instance and then reverses the normalization on the output.
    /// This helps the model handle different scales across time series.
    /// </para>
    /// </remarks>
    public bool UseRevIN { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use batch normalization.
    /// Default: false.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Batch normalization normalizes across the batch dimension.
    /// Usually RevIN is preferred for time series forecasting.
    /// </para>
    /// </remarks>
    public bool UseBatchNorm { get; set; } = false;

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

        if (NumBlocks < 1)
            errors.Add("NumBlocks must be at least 1.");
        if (HiddenDimension < 1)
            errors.Add("HiddenDimension must be at least 1.");
        if (FeedForwardExpansion <= 0)
            errors.Add("FeedForwardExpansion must be positive.");
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
