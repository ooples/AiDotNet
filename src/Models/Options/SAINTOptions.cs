namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the SAINT model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAINT uses attention both across columns (features)
/// and across rows (samples in a batch). These options control how deep and
/// wide that attention mechanism is.
/// </para>
/// </remarks>
public class SAINTOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Higher values let the model learn more complex
    /// patterns but require more compute.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple heads allow the model to focus on
    /// different relationships at the same time.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More layers mean a deeper model that can capture
    /// richer interactions, but it trains more slowly.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Batch size (sequence length for inter-sample attention).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAINT compares rows in a batch to each other.
    /// This value controls how many rows it can compare at once.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// turning off parts of the network during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;
}
