using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TimesNet model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TimesNet transforms 1D time series into 2D tensors based on discovered periods,
/// then applies 2D convolutions to capture both intra-period and inter-period variations.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how the TimesNet model behaves:
///
/// Key settings:
/// - <b>TopK:</b> How many dominant periods to discover from the data
/// - <b>NumLayers:</b> How deep the network is
/// - <b>ConvKernelSize:</b> Size of the 2D convolution kernels
/// - <b>ModelDimension:</b> Size of internal representations
///
/// TimesNet is particularly good at capturing periodic patterns (daily, weekly, seasonal)
/// that are common in financial time series.
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
/// Time Series Analysis", ICLR 2023. https://arxiv.org/abs/2210.02186
/// </para>
/// </remarks>
public class TimesNetOptions<T> : ModelOptions
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of dominant periods to discover.
    /// Default: 5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TimesNet uses FFT to discover the top-K dominant periods in the data.
    /// These periods determine how the 1D time series is reshaped into 2D.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If your data has daily and weekly patterns,
    /// using TopK=5 might discover periods like 1 day, 7 days, 30 days, etc.
    /// </para>
    /// </remarks>
    public int TopK { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of TimesBlock layers.
    /// Default: 2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More layers allow the model to learn more complex patterns but increase
    /// computation time and memory usage.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

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
    /// The FFN dimension in the TimesBlock layers.
    /// Typically 2-4 times the model dimension.
    /// </para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the 2D convolution kernel size.
    /// Default: 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel size determines how much of the 2D
    /// representation is looked at simultaneously. A 3x3 kernel captures
    /// local patterns in both time and period dimensions.
    /// </para>
    /// </remarks>
    public int ConvKernelSize { get; set; } = 3;

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

        if (TopK < 1)
            errors.Add("TopK must be at least 1.");
        if (NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
        if (ModelDimension < 1)
            errors.Add("ModelDimension must be at least 1.");
        if (FeedForwardDimension < 1)
            errors.Add("FeedForwardDimension must be at least 1.");
        if (ConvKernelSize < 1 || ConvKernelSize % 2 == 0)
            errors.Add("ConvKernelSize must be at least 1 and odd.");
        if (Dropout < 0.0 || Dropout >= 1.0)
            errors.Add("Dropout must be in [0, 1).");
        if (LearningRate <= 0)
            errors.Add("LearningRate must be positive.");
        if (SequenceLength < TopK)
            errors.Add("SequenceLength must be at least TopK.");
        if (PredictionHorizon < 1)
            errors.Add("PredictionHorizon must be at least 1.");
        if (NumFeatures < 1)
            errors.Add("NumFeatures must be at least 1.");

        return errors;
    }

    #endregion
}
