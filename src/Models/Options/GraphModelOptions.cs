namespace AiDotNet.Models.Options;

/// <summary>
/// Shared hyperparameters for the graph task models — the node, link and graph-level heads that
/// sit on a message-passing encoder.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models all learn from a graph: a set of items (nodes) joined by
/// relationships (edges). The settings here are the ones every one of them has. Each model's own
/// options class ships the values it was published with, so you do not normally set any of them.
/// </para>
/// <para>
/// The layer count is deliberately NOT here. The three models name it differently
/// (<c>NumLayers</c> for the node and link heads, <c>NumGnnLayers</c> for the graph-level one)
/// and those names are part of each model's published vocabulary, so the property lives on each
/// leaf rather than being renamed to a common word.
/// </para>
/// <para>
/// The maximum gradient norm is not here either — it is already on
/// <see cref="ModelHyperparameterOptions.MaxGradNorm"/>, which every model options class inherits.
/// </para>
/// </remarks>
public abstract class GraphModelOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the width of the hidden message-passing layers. Default: 64.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much information each node can carry while it exchanges
    /// messages with its neighbours. Wider holds more but costs more to train.</para>
    /// </remarks>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate applied between layers. Default: 0.5. Zero disables dropout.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training the model randomly ignores this fraction of its
    /// own internal signals, which stops it leaning too hard on any single one. Graph models use a
    /// notably high rate — 0.5 is standard here, where 0.1 is typical elsewhere.</para>
    /// </remarks>
    public double DropoutRate { get; set; }

    /// <summary>
    /// Throws if a dimension every graph model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options class
    /// did not assign its published defaults.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Only <see cref="HiddenDim"/> is required. <see cref="DropoutRate"/> is deliberately not:
    /// zero legitimately means "no dropout", so requiring it positive would reject a valid
    /// configuration — the over-strictness that made five earlier family bases throw at their own
    /// members' defaults.
    /// </para>
    /// </remarks>
    protected void ValidateCore()
    {
        Require(HiddenDim, nameof(HiddenDim));
    }
}
