using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the QueryMeldNet (MQ-Former) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> QueryMeldNet dynamically melds instance and stuff queries via
/// cross-attention to scale across diverse datasets. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class QueryMeldNetOptions : NeuralNetworkOptions
{
}
