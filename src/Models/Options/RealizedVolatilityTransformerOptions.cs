namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Realized Volatility Transformer model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how the transformer looks back in time
/// and how large the attention layers are when predicting volatility.
/// </para>
/// </remarks>
public class RealizedVolatilityTransformerOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of assets modeled at once.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set this to the number of stocks or assets you track.</para>
    /// </remarks>
    public int NumAssets { get; set; } = 1;

    /// <summary>
    /// Gets or sets the lookback window (number of past time steps).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A lookback of 90 means the model sees the last 90 returns.</para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 90;

    /// <summary>
    /// Gets or sets the forecast horizon (number of future steps).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A horizon of 5 forecasts volatility for the next 5 periods.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 5;

    /// <summary>
    /// Gets or sets the transformer hidden dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger hidden sizes let the model learn richer patterns.</para>
    /// </remarks>
    public int HiddenSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More heads let the model focus on multiple time patterns at once.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers can capture longer-term dependencies.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent the model from memorizing noise.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the Adam learning rate used when no optimizer is supplied to the constructor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Defaults to 3e-4 rather than Adam's stock 1e-3. Adam's first update is very close to
    /// sign(gradient) * learningRate on every parameter at once, and at 1e-3 that coherent step
    /// is large enough on a 128-wide, 2-block encoder to throw the forecast far past the target
    /// before the moment estimates settle. Measured over eight random initializations at ten
    /// steps each, 1e-3 left the loss above where it started once in eight runs while 3e-4
    /// reduced it every time and to the lowest value of any rate tried.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Pass your
    /// own optimizer to the constructor if you want a different rule entirely; this only sets the
    /// rate of the default one.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>
    /// Validates the option values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks that the settings are positive and sensible.</para>
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
        if (NumHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(NumHeads), "NumHeads must be at least 1.");
        if (NumLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(NumLayers), "NumLayers must be at least 1.");
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), "DropoutRate must be in [0, 1).");
        if (double.IsNaN(LearningRate) || double.IsInfinity(LearningRate) || LearningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), "LearningRate must be finite and positive.");
    }
}
