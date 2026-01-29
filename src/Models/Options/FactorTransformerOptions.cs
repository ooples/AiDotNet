namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the FactorTransformer model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FactorTransformer uses attention to learn cross-sectional and temporal relationships.
/// These options define the transformer depth, head count, and factor dimensions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like tuning a transformer for markets:
/// - More heads let the model look at multiple relationships at once
/// - More layers allow deeper reasoning but cost more compute
/// </para>
/// </remarks>
public class FactorTransformerOptions<T>
{
    /// <summary>
    /// Gets or sets the number of latent factors to learn.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the number of hidden drivers the model discovers.
    /// </para>
    /// </remarks>
    public int NumFactors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of assets covered by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls the size of the asset output layer.
    /// </para>
    /// </remarks>
    public int NumAssets { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Features can include returns, technical indicators, and fundamentals.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 50;

    /// <summary>
    /// Gets or sets the transformer hidden dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the internal representation used by attention.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each head learns different relationships in the data.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More layers allow deeper reasoning but increase training time.
    /// </para>
    /// </remarks>
    public int NumTransformerLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps of history the model processes at once.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 60;

    /// <summary>
    /// Gets or sets the prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many steps ahead the model predicts.
    /// </para>
    /// </remarks>
    public int PredictionHorizon { get; set; } = 20;

    /// <summary>
    /// Gets or sets the dropout rate used for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout helps prevent overfitting by randomly disabling neurons.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Validates the options and throws if any value is invalid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This prevents impossible settings (like zero heads or negative sizes).
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
        if (NumHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumHeads), "NumHeads must be positive.");
        if (NumTransformerLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumTransformerLayers), "NumTransformerLayers must be positive.");
        if (SequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(SequenceLength), "SequenceLength must be positive.");
        if (PredictionHorizon <= 0)
            throw new ArgumentOutOfRangeException(nameof(PredictionHorizon), "PredictionHorizon must be positive.");
        if (HiddenDimension % NumHeads != 0)
            throw new ArgumentException("HiddenDimension must be divisible by NumHeads.", nameof(HiddenDimension));
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), "DropoutRate must be in [0, 1).");
    }
}
