namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the AlphaFactorModel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AlphaFactorModel learns latent factors that explain and predict excess returns.
/// These options let you control the factor count, input dimensionality, and hidden size.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of these settings as the knobs that shape the model:
/// - How many hidden factors to learn
/// - How many assets and features are in your data
/// - How large the internal layers should be
/// </para>
/// </remarks>
public class AlphaFactorOptions<T>
{
    /// <summary>
    /// Gets or sets the number of latent factors to learn.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many hidden "drivers" of returns the model tries to discover.
    /// More factors can capture more nuance but may overfit if data is limited.
    /// </para>
    /// </remarks>
    public int NumFactors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of assets covered by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you model the S&amp;P 500, this would be 500.
    /// It controls the size of the output layer for predicted excess returns.
    /// </para>
    /// </remarks>
    public int NumAssets { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of input features per asset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Features can be prices, volumes, ratios, or indicators.
    /// More features provide richer input but also increase model complexity.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 50;

    /// <summary>
    /// Gets or sets the width of hidden layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Larger hidden layers can model more complex patterns,
    /// but require more data and computation to train well.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps of history the model sees at once.
    /// For daily data, a value of 60 means about three months of history.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 60;

    /// <summary>
    /// Gets or sets the prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps ahead the model predicts.
    /// A value of 20 could mean predicting about one month ahead for daily data.
    /// </para>
    /// </remarks>
    public int PredictionHorizon { get; set; } = 20;

    /// <summary>
    /// Gets or sets the dropout rate used for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly disables some neurons during training.
    /// This helps prevent overfitting by forcing the model to learn more robust patterns.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Validates the options and throws if any value is invalid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a safety check that catches impossible settings
    /// (like negative dimensions) before training starts.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (NumFactors <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumFactors), "NumFactors must be positive.");
        if (NumAssets <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumAssets), "NumAssets must be positive.");
        if (NumFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumFeatures), "NumFeatures must be positive.");
        if (HiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenDimension), "HiddenDimension must be positive.");
        if (SequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(SequenceLength), "SequenceLength must be positive.");
        if (PredictionHorizon <= 0)
            throw new ArgumentOutOfRangeException(nameof(PredictionHorizon), "PredictionHorizon must be positive.");
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), "DropoutRate must be in [0, 1).");
    }
}
