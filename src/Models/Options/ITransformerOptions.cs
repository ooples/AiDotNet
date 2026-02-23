using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the iTransformer (Inverted Transformer) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// iTransformer inverts the traditional transformer approach by treating each variable (channel)
/// as a token instead of each time step. This allows the model to learn cross-variable dependencies
/// more effectively.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional transformers process time series by treating each time step
/// as a "word" (token). iTransformer flips this - it treats each variable (like price, volume, etc.)
/// as a token. This helps the model learn how different variables relate to each other.
///
/// Key settings:
/// - <b>NumLayers:</b> How many transformer layers to stack. More layers = more capacity.
/// - <b>NumHeads:</b> Number of attention heads. Each head learns different relationships.
/// - <b>ModelDimension:</b> Size of internal representations. Larger = more expressive.
/// - <b>UseChannelAttention:</b> If true, applies attention across channels (the key innovation).
///
/// Default values are from the original iTransformer paper and work well for most datasets.
/// </para>
/// <para>
/// <b>Reference:</b> Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
/// ICLR 2024. https://arxiv.org/abs/2310.06625
/// </para>
/// </remarks>
public class ITransformerOptions<T> : ModelOptions
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// Default: 2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each layer learns increasingly abstract patterns from the data.
    /// More layers can capture more complex relationships but require more computation.
    /// The iTransformer paper found 2 layers works well for most forecasting tasks.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// Default: 8.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple attention heads allow the model to focus on different
    /// aspects of the relationships between variables simultaneously. For example, one head
    /// might learn price-volume correlations while another learns momentum patterns.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the model dimension (embedding size).
    /// Default: 512.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This determines how much information each variable embedding
    /// can hold. Larger dimensions can represent more nuanced patterns but use more memory.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the feedforward network dimension.
    /// Default: 512.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The FFN processes each token after attention. This dimension
    /// controls its capacity. Often set equal to or larger than the model dimension.
    /// </para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 512;

    #endregion

    #region Attention Configuration

    /// <summary>
    /// Gets or sets whether to use channel attention (inverted attention).
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the key innovation of iTransformer. When true,
    /// attention is computed across variables/channels instead of across time steps.
    /// This helps the model learn how different variables (like OHLCV data) relate to each other.
    /// </para>
    /// </remarks>
    public bool UseChannelAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// Default: 0.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly zeros out attention weights during training
    /// to prevent overfitting. 0.0 means no dropout (as used in the original paper).
    /// </para>
    /// </remarks>
    public double AttentionDropout { get; set; } = 0.0;

    #endregion

    #region Regularization

    /// <summary>
    /// Gets or sets the dropout rate.
    /// Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly disables neurons during training to prevent
    /// overfitting. A rate of 0.1 means 10% of neurons are randomly turned off each pass.
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
    /// <b>For Beginners:</b> RevIN normalizes each input sequence to handle distribution shift
    /// (when data statistics change over time, like stock prices going from $100 to $1000).
    /// Highly recommended for financial time series.
    /// </para>
    /// </remarks>
    public bool UseInstanceNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pre-normalization (LayerNorm before attention).
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pre-norm applies layer normalization before attention and FFN layers,
    /// which typically leads to more stable training compared to post-norm.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many historical time steps the model looks at.
    /// For hourly data, 96 means looking at the last 4 days of history.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 96;

    /// <summary>
    /// Gets or sets the prediction horizon.
    /// Default: 96.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many future time steps to predict.
    /// iTransformer can handle longer horizons than patch-based models.
    /// </para>
    /// </remarks>
    public int PredictionHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the number of input features (variables/channels).
    /// Default: 7.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The number of different variables in your time series.
    /// For stock data, this might be Open, High, Low, Close, Volume, and derived indicators.
    /// In iTransformer, each feature becomes a token that attends to other features.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 7;

    #endregion

    #region Validation

    /// <summary>
    /// Validates the options and returns any validation errors.
    /// </summary>
    /// <returns>List of validation error messages, empty if valid.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks that all your configuration settings are valid
    /// before creating the model. It catches common mistakes like setting dimensions to zero
    /// or using incompatible values.
    /// </para>
    /// </remarks>
    public List<string> Validate()
    {
        var errors = new List<string>();

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
        if (Dropout < 0.0 || Dropout >= 1.0)
            errors.Add("Dropout must be in [0, 1).");
        if (AttentionDropout < 0.0 || AttentionDropout >= 1.0)
            errors.Add("AttentionDropout must be in [0, 1).");
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
