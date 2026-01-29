namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TabTransformer model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how the transformer processes
/// categorical fields (like sector, exchange, or country) and mixes them with
/// numeric features (like price or volume).
/// </para>
/// </remarks>
public class TabTransformerOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the width of the internal representations.
    /// Larger values can capture more complex patterns but cost more to train.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple heads let the model focus on different
    /// relationships at the same time.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More layers means a deeper model that can learn
    /// richer patterns, but it is slower to train.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Number of categorical features (determines sequence length for transformer).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many categorical columns you have.
    /// TabTransformer uses this as the sequence length for attention.
    /// </para>
    /// </remarks>
    public int NumCategoricalFeatures { get; set; } = 10;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// turning off some neurons during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;
}
