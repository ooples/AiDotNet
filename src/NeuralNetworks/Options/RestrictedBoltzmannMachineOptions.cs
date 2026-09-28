using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the RestrictedBoltzmannMachine.
/// </summary>
public class RestrictedBoltzmannMachineOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets cd steps. Default: <c>1</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many back-and-forth sampling steps are used to estimate the gradient. One works well in practice.</para>
    /// </remarks>
    public int CdSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets learning rate. Default: <c>0.01</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes when it corrects itself.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(CdSteps, nameof(CdSteps));
        Require(LearningRate, nameof(LearningRate));
    }
}
