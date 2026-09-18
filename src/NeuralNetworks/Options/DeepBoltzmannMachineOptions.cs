using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the DeepBoltzmannMachine.
/// </summary>
public class DeepBoltzmannMachineOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets batch size. Default: <c>32</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many examples the model looks at before it updates itself once.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets cd steps. Default: <c>1</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many back-and-forth sampling steps are used to estimate the gradient. One works well in practice.</para>
    /// </remarks>
    public int CdSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets epochs. Default: <c>10</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many complete passes the model makes over your training data.</para>
    /// </remarks>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets learning rate decay. Default: <c>1.0</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much the step size shrinks each epoch. 1.0 means no shrinking.</para>
    /// </remarks>
    public double LearningRateDecay { get; set; } = 1.0;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(BatchSize, nameof(BatchSize));
        Require(CdSteps, nameof(CdSteps));
        Require(Epochs, nameof(Epochs));
        Require(LearningRateDecay, nameof(LearningRateDecay));
    }
}
