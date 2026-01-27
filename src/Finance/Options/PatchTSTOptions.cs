using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Options;

/// <summary>
/// Configuration options for the PatchTST (Patch Time Series Transformer) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// PatchTST divides time series into patches and processes them with a Transformer encoder.
/// This approach has shown state-of-the-art results on long-term forecasting benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how the PatchTST model behaves:
///
/// Key settings:
/// - <b>PatchSize:</b> How long each "patch" (segment) of time series is.
///   Smaller patches capture fine details; larger patches capture broader patterns.
/// - <b>Stride:</b> How much overlap between patches. Equal to PatchSize for no overlap.
/// - <b>NumLayers:</b> How deep the transformer is. More layers = more capacity but slower.
/// - <b>NumHeads:</b> Attention heads. More heads can capture different types of patterns.
/// - <b>ChannelIndependent:</b> Whether to process each variable independently.
///   Usually True works better for multivariate forecasting.
///
/// Default values are from the original PatchTST paper and work well for most datasets.
/// </para>
/// <para>
/// <b>Reference:</b> Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
/// with Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730
/// </para>
/// </remarks>
public class PatchTSTOptions<T>
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the patch size (segment length).
    /// Default: 16.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The patch size determines how the input sequence is divided into segments.
    /// Each patch becomes a "token" that the transformer processes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If you have 96 time steps and patch size is 16,
    /// the model creates 6 patches (96 / 16 = 6). Each patch represents 16 consecutive
    /// time steps, compressed into a single "word" the transformer can understand.
    /// </para>
    /// </remarks>
    public int PatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the stride between consecutive patches.
    /// Default: 8 (overlapping patches).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A stride smaller than patch size creates overlapping patches, which can improve
    /// the model's ability to capture patterns that span patch boundaries.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> With patch size 16 and stride 8, consecutive patches
    /// share 8 time steps. This overlap helps the model not "miss" patterns that happen
    /// at the boundary between patches.
    /// </para>
    /// </remarks>
    public int Stride { get; set; } = 8;

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
    /// the sequence simultaneously, capturing various types of temporal patterns.
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

    #endregion

    #region Channel Configuration

    /// <summary>
    /// Gets or sets whether to use channel-independent (CI) mode.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In CI mode, each channel (variable) is processed independently through the same
    /// model, sharing parameters across channels. This often improves generalization
    /// for multivariate time series forecasting.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If you have 7 variables (like different stock features),
    /// CI mode processes each one separately using the same weights. This helps the model
    /// learn patterns that apply to all variables rather than memorizing variable-specific quirks.
    /// </para>
    /// </remarks>
    public bool ChannelIndependent { get; set; } = true;

    #endregion

    #region Regularization

    /// <summary>
    /// Gets or sets the dropout rate.
    /// Default: 0.05.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout randomly zeros out a fraction of neurons during training,
    /// helping prevent overfitting.
    /// </para>
    /// </remarks>
    public double Dropout { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// Default: 0.0 (no attention dropout).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout applied to attention weights. The PatchTST paper uses 0.0.
    /// </para>
    /// </remarks>
    public double AttentionDropout { get; set; } = 0.0;

    #endregion

    #region Normalization

    /// <summary>
    /// Gets or sets whether to use instance normalization (RevIN).
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instance normalization (RevIN) helps handle distribution shift in time series data
    /// by normalizing each instance to zero mean and unit variance, then denormalizing
    /// the predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Time series data often has changing statistics over time
    /// (e.g., stock prices going from $100 to $500). RevIN normalizes each input sequence
    /// so the model sees standardized data, then "un-normalizes" the output predictions.
    /// </para>
    /// </remarks>
    public bool UseInstanceNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pre-norm (LayerNorm before attention/FFN) or post-norm.
    /// Default: true (pre-norm, recommended).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Pre-norm applies layer normalization before the attention and feedforward layers,
    /// which typically leads to more stable training.
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

        if (PatchSize < 1)
            errors.Add("PatchSize must be at least 1.");
        if (Stride < 1)
            errors.Add("Stride must be at least 1.");
        if (Stride > PatchSize)
            errors.Add("Stride cannot be greater than PatchSize.");
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
        if (SequenceLength < PatchSize)
            errors.Add("SequenceLength must be at least PatchSize.");
        if (PredictionHorizon < 1)
            errors.Add("PredictionHorizon must be at least 1.");
        if (NumFeatures < 1)
            errors.Add("NumFeatures must be at least 1.");

        return errors;
    }

    /// <summary>
    /// Calculates the number of patches based on sequence length, patch size, and stride.
    /// </summary>
    /// <returns>The number of patches.</returns>
    public int CalculateNumPatches()
    {
        return (SequenceLength - PatchSize) / Stride + 1;
    }

    #endregion
}
