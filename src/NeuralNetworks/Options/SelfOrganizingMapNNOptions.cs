using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SelfOrganizingMap neural network.
/// </summary>
public class SelfOrganizingMapNNOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets the training length T in steps (one step per <c>Train</c> call): the learning rate falls linearly
    /// to zero and the neighborhood radius to one over T steps, as in SOM_PAK. Default: <c>1000</c>.
    /// </summary>
    public int TotalEpochs { get; set; } = 1000;

    /// <summary>Gets or sets the learning rate α(0) at the first step. Default: <c>0.1</c>.</summary>
    public double InitialLearningRate { get; set; } = 0.1;

    /// <summary>Gets or sets the neighborhood radius at the first step; non-positive means half the larger grid side.
    /// Default: <c>-1</c>.</summary>
    public double InitialRadius { get; set; } = -1;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(TotalEpochs, nameof(TotalEpochs));
    }
}
