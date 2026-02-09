using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Base options for risk models.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class RiskModelOptions<T> : FinancialNeuralNetworkOptions
{
    /// <summary>
    /// Number of input features used for risk calculation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many columns (inputs) the model expects.
    /// If your data has 10 indicators (price, volume, volatility, etc.), set this to 10.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 10;

    /// <summary>
    /// Confidence level for risk metrics (e.g., 0.95, 0.99).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> 0.95 means "I want to be safe 95% of the time."
    /// Higher values are more conservative but may reduce returns.
    /// </para>
    /// </remarks>
    public double ConfidenceLevel { get; set; } = 0.95;

    /// <summary>
    /// Time horizon for risk estimation in days.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how far into the future the risk is measured.
    /// A value of 1 means "tomorrow," while 10 means "the next 10 days."
    /// </para>
    /// </remarks>
    public int TimeHorizon { get; set; } = 1;

    /// <summary>
    /// Optional loss function override for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss function tells the model how wrong it is during training.
    /// Leave this null to use a sensible default (usually Mean Squared Error).
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; }
}
