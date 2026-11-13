namespace AiDotNet.Models.Options;

/// <summary>
/// Base configuration options for financial neural network models.
/// </summary>
public class FinancialNeuralNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>Initializes a new instance with the default settings.</summary>
    public FinancialNeuralNetworkOptions()
    {
    }

    /// <summary>Initializes a new instance by copying every setting of another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    protected FinancialNeuralNetworkOptions(FinancialNeuralNetworkOptions other) : base(other)
    {
    }
}
