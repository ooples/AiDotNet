using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Represents a model partitioned for cloud+edge deployment.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PartitionedModel provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class PartitionedModel<T, TInput, TOutput>
{
    /// <summary>Gets or sets the original model.</summary>
    public IFullModel<T, TInput, TOutput>? OriginalModel { get; set; }

    /// <summary>Gets or sets the model part for edge execution.</summary>
    public IFullModel<T, TInput, TOutput>? EdgeModel { get; set; }

    /// <summary>Gets or sets the model part for cloud execution.</summary>
    public IFullModel<T, TInput, TOutput>? CloudModel { get; set; }

    /// <summary>Gets or sets the partition strategy used.</summary>
    public PartitionStrategy PartitionStrategy { get; set; }

    /// <summary>Gets or sets the intermediate tensor shape between edge and cloud.</summary>
    public int[]? IntermediateShape { get; set; }
}
