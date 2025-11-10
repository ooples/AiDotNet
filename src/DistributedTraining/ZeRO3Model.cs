using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 3 model wrapper - full sharding of parameters, gradients, and optimizer states.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// ZeRO Stage 3 is the full implementation of the ZeRO optimization, sharding parameters, gradients,
/// AND optimizer states across all processes. This is equivalent to PyTorch's FSDP (Fully Sharded Data Parallel).
/// Parameters are gathered just-in-time for forward/backward passes and immediately released,
/// maximizing memory efficiency.
/// </para>
/// <para><b>For Beginners:</b>
/// ZeRO-3 is identical to FSDP - it's the ultimate memory-saving strategy. Everything is sharded:
/// parameters, gradients, and optimizer states. Each process only holds a small piece of the model,
/// and pieces are gathered only when absolutely needed, then immediately released.
/// </para>
/// <para>
/// This class is essentially an alias/wrapper for FSDPModel to maintain ZeRO naming consistency.
/// </para>
/// <para><b>Use Cases:</b>
/// - Same as FSDP - training very large models
/// - When you prefer ZeRO terminology over FSDP
/// - Maximum memory efficiency
/// </para>
/// <para><b>Trade-offs:</b>
/// - Same as FSDP
/// - Memory: Excellent - everything sharded
/// - Communication: Higher - AllGather for each forward/backward
/// - Complexity: Moderate
/// </para>
/// <para>
/// Example:
/// <code>
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// // ZeRO-3 and FSDP are equivalent
/// var zero3Model = new ZeRO3Model&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
/// // Or equivalently:
/// // var fsdpModel = new FSDPModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO3Model<T, TInput, TOutput> : FSDPModel<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a new ZeRO-3 model wrapping an existing model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// ZeRO-3 is the same as FSDP, just different terminology. Use whichever name you prefer.
    /// This constructor delegates to FSDPModel for all functionality.
    /// </para>
    /// </remarks>
    /// <param name="wrappedModel">The model to wrap with ZeRO-3 capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    public ZeRO3Model(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();

        // Update strategy name to reflect ZeRO-3
        metadata.SetProperty("Strategy", "ZeRO-3");
        metadata.SetProperty("EquivalentTo", "FSDP");
        metadata.SetProperty("ParametersSharded", true);
        metadata.SetProperty("GradientsSharded", true);
        metadata.SetProperty("OptimizerStateSharded", true);

        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = WrappedModel.WithParameters(parameters);
        return new ZeRO3Model<T, TInput, TOutput>(newModel, Config);
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> Clone()
    {
        var clonedWrappedModel = WrappedModel.Clone();
        return new ZeRO3Model<T, TInput, TOutput>(clonedWrappedModel, Config);
    }
}
