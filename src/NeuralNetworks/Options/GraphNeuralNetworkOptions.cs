using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GraphNeuralNetwork.
/// </summary>
public class GraphNeuralNetworkOptions : NeuralNetworkOptions
{
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

    /// <summary>Gets or sets whether the optional graph-smoothness loss is enabled.</summary>
    /// <remarks>Disabled by default because it is not part of the original GCN objective.</remarks>
    public bool UseAuxiliaryLoss { get; set; }

    /// <summary>Gets or sets the optional graph-smoothness loss weight.</summary>
    public double AuxiliaryLossWeight { get; set; } = 0.05;
}
