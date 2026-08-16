using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GraphNeuralNetwork.
/// </summary>
public class GraphNeuralNetworkOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GraphNeuralNetworkOptions() { }

    /// <summary>Initializes a new instance by copying every property from another instance.</summary>
    /// <param name="other">The instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    /// <remarks>
    /// This lives here, next to the properties, so a property added below is added a line away from
    /// the place that has to copy it. GraphNeuralNetwork used to enumerate the list by hand in a
    /// private CopyOptions, where a new property was simply dropped from every clone -- the clone
    /// silently reverting to the default while the original kept the configured value.
    /// Seed comes from the base and is copied for the same reason: it is not declared in this file.
    /// </remarks>
    public GraphNeuralNetworkOptions(GraphNeuralNetworkOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        NodeFeatureSize = other.NodeFeatureSize;
        NumClasses = other.NumClasses;
        HiddenSize = other.HiddenSize;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        L2Regularization = other.L2Regularization;
        UseBias = other.UseBias;
        UseAuxiliaryLoss = other.UseAuxiliaryLoss;
        AuxiliaryLossWeight = other.AuxiliaryLossWeight;
    }

    /// <summary>Gets or sets the number of input features per node.</summary>
    /// <remarks>The default of 1433 matches the paper's Cora configuration.</remarks>
    public int NodeFeatureSize { get; set; } = 1433;

    /// <summary>Gets or sets the number of output classes.</summary>
    /// <remarks>The default of 7 matches the Cora dataset.</remarks>
    public int NumClasses { get; set; } = 7;

    /// <summary>Gets or sets the hidden feature width.</summary>
    /// <remarks>The original GCN uses 16 hidden units.</remarks>
    public int HiddenSize { get; set; } = 16;

    /// <summary>Gets or sets the total number of graph-convolutional layers.</summary>
    /// <remarks>The original GCN uses two graph-convolutional layers.</remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>Gets or sets the dropout probability.</summary>
    /// <remarks>The paper uses a dropout rate of 0.5.</remarks>
    public double DropoutRate { get; set; } = 0.5;

    /// <summary>Gets or sets the Adam learning rate.</summary>
    /// <remarks>The paper uses a learning rate of 0.01.</remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>Gets or sets the L2 regularization strength.</summary>
    /// <remarks>The paper uses 5e-4 weight decay.</remarks>
    public double L2Regularization { get; set; } = 5e-4;

    /// <summary>Gets or sets whether graph-convolutional layers include trainable biases.</summary>
    /// <remarks>
    /// Disabled by default because the original GCN reference implementation constructs both
    /// graph-convolutional layers with <c>bias=False</c>. Enable this for bias-augmented variants.
    /// </remarks>
    public bool UseBias { get; set; }

    /// <summary>Gets or sets whether the optional graph-smoothness loss is enabled.</summary>
    /// <remarks>Disabled by default because it is not part of the original GCN objective.</remarks>
    public bool UseAuxiliaryLoss { get; set; }

    /// <summary>Gets or sets the optional graph-smoothness loss weight.</summary>
    public double AuxiliaryLossWeight { get; set; } = 0.05;
}
