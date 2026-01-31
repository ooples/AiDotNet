namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Neural GARCH volatility model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how much history the model sees,
/// how big the neural network is, and how far ahead you want to forecast volatility.
/// </para>
/// </remarks>
public class NeuralGARCHOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of assets modeled at once.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you are modeling 5 stocks together, set this to 5.</para>
    /// </remarks>
    public int NumAssets { get; set; } = 1;

    /// <summary>
    /// Gets or sets the lookback window (number of past time steps).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A lookback of 60 means the model looks at the last 60 returns.</para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 60;

    /// <summary>
    /// Gets or sets the forecast horizon (number of future steps).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A horizon of 1 forecasts volatility for the next period only.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden layer width.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger hidden sizes let the model learn more complex patterns.</para>
    /// </remarks>
    public int HiddenSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers can capture deeper patterns but need more data.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly turns off neurons during training to reduce overfitting.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Validates the option values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks that the settings make sense (no negative sizes).</para>
    /// </remarks>
    public void Validate()
    {
        if (NumAssets < 1)
            throw new ArgumentOutOfRangeException(nameof(NumAssets), "NumAssets must be at least 1.");
        if (LookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(LookbackWindow), "LookbackWindow must be at least 1.");
        if (ForecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(ForecastHorizon), "ForecastHorizon must be at least 1.");
        if (HiddenSize < 1)
            throw new ArgumentOutOfRangeException(nameof(HiddenSize), "HiddenSize must be at least 1.");
        if (NumLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(NumLayers), "NumLayers must be at least 1.");
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), "DropoutRate must be in [0, 1).");
    }
}
