using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the FEDformer (Frequency Enhanced Decomposed Transformer) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// FEDformer achieves linear complexity by performing attention in the frequency domain
/// using Fourier or Wavelet transforms. It also uses seasonal-trend decomposition for
/// better interpretability.
/// </para>
/// <para>
/// <b>For Beginners:</b> FEDformer is like listening to music - instead of processing each
/// sound wave individually (time domain), it analyzes the frequencies (like bass and treble).
/// This makes it much faster while still capturing important patterns.
///
/// Key innovations:
/// - <b>Frequency Attention:</b> Computes attention in frequency domain (O(n) vs O(n²))
/// - <b>Decomposition:</b> Separates trend (overall direction) from seasonal (repeating patterns)
/// - <b>Random Selection:</b> Randomly samples frequencies for efficiency
///
/// Default values are from the original FEDformer paper and work well for most datasets.
/// </para>
/// <para>
/// <b>Reference:</b> Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
/// ICML 2022. https://arxiv.org/abs/2201.12740
/// </para>
/// </remarks>
public class FEDformerOptions<T> : ModelOptions
{
    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public FEDformerOptions() { }

    /// <summary>
    /// Creates a copy of the specified options.
    /// </summary>
    /// <param name="other">The options to copy.</param>
    public FEDformerOptions(FEDformerOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so it is copied explicitly: losing it on a clone
        // silently changes deterministic initialization.
        Seed = other.Seed;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        ModelDimension = other.ModelDimension;
        NumHeads = other.NumHeads;
        FeedForwardDimension = other.FeedForwardDimension;
        AttentionType = other.AttentionType;
        NumModes = other.NumModes;
        ModeSelection = other.ModeSelection;
        ModeSelectionSeed = other.ModeSelectionSeed;
        MovingAverageKernel = other.MovingAverageKernel;
        Dropout = other.Dropout;
        UseInstanceNormalization = other.UseInstanceNormalization;
        LossFunction = other.LossFunction;
        LearningRate = other.LearningRate;
        SequenceLength = other.SequenceLength;
        LabelLength = other.LabelLength;
        PredictionHorizon = other.PredictionHorizon;
        NumFeatures = other.NumFeatures;
    }

    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// Default: 2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Encoder layers process the input sequence to understand patterns.
    /// More layers = more capacity to learn complex patterns, but slower training.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// Default: 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decoder layers generate the predictions. FEDformer uses fewer
    /// decoder layers than encoder layers because the decomposition already simplifies the task.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the model dimension.
    /// Default: 512.
    /// </summary>
    public int ModelDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// Default: 8.
    /// </summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the feedforward dimension.
    /// Default: 2048.
    /// </summary>
    public int FeedForwardDimension { get; set; } = 2048;

    #endregion

    #region Frequency Attention Configuration

    /// <summary>
    /// Gets or sets the frequency attention type.
    /// Default: Fourier.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FEDformer can use two types of frequency transforms:
    /// - <b>Fourier:</b> Classic frequency analysis, good for periodic patterns
    /// - <b>Wavelet:</b> Better for patterns that change over time (localized frequency)
    /// </para>
    /// </remarks>
    public FrequencyAttentionType AttentionType { get; set; } = FrequencyAttentionType.Fourier;

    /// <summary>
    /// Gets or sets the number of frequency modes to keep.
    /// Default: 64.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of using all frequencies, FEDformer randomly selects
    /// a subset. This makes computation faster while keeping the most important information.
    /// Think of it like compressing music - you keep the frequencies humans can hear best.
    /// </para>
    /// </remarks>
    public int NumModes { get; set; } = 64;

    /// <summary>
    /// Gets or sets how the frequency modes are chosen.
    /// Default: <see cref="FourierModeSelection.Random"/>, the paper's choice.
    /// </summary>
    /// <value>The rule that picks which <see cref="NumModes"/> frequencies each Fourier block keeps.</value>
    /// <remarks>
    /// <para>
    /// Zhou et al. (Sec. 3.2) select the modes at random once, when the model is built: random selection keeps some
    /// high frequencies that a low-pass choice would drop. <see cref="FourierModeSelection.Lowest"/> keeps the lowest
    /// <see cref="NumModes"/> frequencies instead.
    /// </para>
    /// <para><b>For Beginners:</b> FEDformer looks at only some of a series' frequencies. Random (the default, as in
    /// the paper) mixes slow and fast ones, which tends to keep more of the series' structure; Lowest keeps only the
    /// slow, smooth ones.</para>
    /// </remarks>
    public FourierModeSelection ModeSelection { get; set; } = FourierModeSelection.Random;

    /// <summary>
    /// Gets or sets the seed of the random mode selection.
    /// Default: 0.
    /// </summary>
    /// <value>The seed that fixes which frequencies <see cref="FourierModeSelection.Random"/> picks.</value>
    /// <remarks>
    /// <para>
    /// The selected modes are part of the network, like its weights, so they come from this seed rather than from the
    /// run's randomness: a copy or a reloaded model reads the same frequencies.
    /// </para>
    /// <para><b>For Beginners:</b> Two models with the same seed look at the same frequencies, so their weights mean
    /// the same thing; change it to try another random choice of frequencies.</para>
    /// </remarks>
    public int ModeSelectionSeed { get; set; } = 0;

    /// <summary>
    /// Gets or sets the moving average kernel size for trend extraction.
    /// Default: 24, the official FEDformer configuration's.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The decomposition separates trend from seasonal patterns using
    /// a moving average. This kernel size determines how smooth the trend is - larger values
    /// give smoother trends.
    /// </para>
    /// </remarks>
    public int MovingAverageKernel { get; set; } = 24;

    #endregion

    #region Regularization

    /// <summary>
    /// Gets or sets the dropout rate.
    /// Default: 0.05.
    /// </summary>
    public double Dropout { get; set; } = 0.05;

    #endregion

    #region Normalization

    /// <summary>
    /// Gets or sets whether to wrap the network in instance normalization (RevIN, Kim et al. 2022).
    /// Default: false: FEDformer (Zhou et al. 2022) has none; its decomposition handles the level.
    /// </summary>
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

    #endregion

    #region Input/Output Configuration

    /// <summary>
    /// Gets or sets the input sequence length (lookback window).
    /// Default: 96.
    /// </summary>
    public int SequenceLength { get; set; } = 96;

    /// <summary>
    /// Gets or sets the label length (overlap between input and prediction).
    /// Default: 48.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FEDformer uses an encoder-decoder architecture where the decoder
    /// sees some of the input sequence. LabelLength determines how much overlap there is.
    /// This helps the decoder have context for generating predictions.
    /// </para>
    /// </remarks>
    public int LabelLength { get; set; } = 48;

    /// <summary>
    /// Gets or sets the prediction horizon.
    /// Default: 96.
    /// </summary>
    public int PredictionHorizon { get; set; } = 96;

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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks your configuration before creating the model,
    /// catching common mistakes early.
    /// </para>
    /// </remarks>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (NumEncoderLayers < 1)
            errors.Add("NumEncoderLayers must be at least 1.");
        if (NumDecoderLayers < 1)
            errors.Add("NumDecoderLayers must be at least 1.");
        if (ModelDimension < 1)
            errors.Add("ModelDimension must be at least 1.");
        if (NumHeads < 1)
            errors.Add("NumHeads must be at least 1.");
        if (ModelDimension % NumHeads != 0)
            errors.Add("ModelDimension must be divisible by NumHeads.");
        if (FeedForwardDimension < 1)
            errors.Add("FeedForwardDimension must be at least 1.");
        if (NumModes < 1)
            errors.Add("NumModes must be at least 1.");
        if (MovingAverageKernel < 1)
            errors.Add("MovingAverageKernel must be at least 1.");
        if (Dropout < 0.0 || Dropout >= 1.0)
            errors.Add("Dropout must be in [0, 1).");
        if (LearningRate <= 0)
            errors.Add("LearningRate must be positive.");
        if (SequenceLength < 1)
            errors.Add("SequenceLength must be at least 1.");
        if (LabelLength < 0 || LabelLength > SequenceLength)
            errors.Add("LabelLength must be between 0 and SequenceLength.");
        if (PredictionHorizon < 1)
            errors.Add("PredictionHorizon must be at least 1.");
        if (NumFeatures < 1)
            errors.Add("NumFeatures must be at least 1.");

        return errors;
    }

    #endregion
}

/// <summary>
/// Specifies the type of frequency attention to use in FEDformer.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These are two different ways to analyze frequency content:
/// - Fourier is simpler and works well for stationary patterns
/// - Wavelet can handle patterns that change over time better
/// </para>
/// </remarks>
public enum FrequencyAttentionType
{
    /// <summary>
    /// Uses Fourier transform for frequency attention.
    /// Best for data with regular, periodic patterns.
    /// </summary>
    Fourier,

    /// <summary>
    /// Uses Wavelet transform for frequency attention.
    /// Best for data with patterns that change over time.
    /// </summary>
    Wavelet
}

/// <summary>
/// How FEDformer chooses the frequency modes its Fourier blocks keep.
/// </summary>
public enum FourierModeSelection
{
    /// <summary>A random subset of the frequencies below the Nyquist frequency, drawn once (the paper's choice).</summary>
    Random,

    /// <summary>The lowest frequencies.</summary>
    Lowest
}
