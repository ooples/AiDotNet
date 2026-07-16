using AiDotNet.Autodiff;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Megatron-LM's <c>f</c> conjugate operator at the INPUT of a tensor-parallel region
/// (Shoeybi et al. 2019, §3): identity in the forward pass, all-reduce in the backward pass.
/// </summary>
/// <remarks>
/// A column-parallel linear replicates its input across the tensor-parallel ranks and each rank
/// produces a distinct slice of the output, so the true gradient of the (shared) input is the SUM
/// of every rank's local input-gradient. Placing this op at the region input makes the forward a
/// no-op while its tape backward all-reduces the incoming gradient, giving <c>dX = Σ_r dX_r</c>
/// without the layer needing a manual backward (the framework is tape-autodiff only).
/// </remarks>
internal sealed class CopyToTensorParallelRegion<T> : AutogradFunction<T>
{
    private readonly ICommunicationBackend<T> _backend;
    public CopyToTensorParallelRegion(ICommunicationBackend<T> backend) { _backend = backend; }

    public override Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs) => inputs[0];

    public override Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput)
    {
        if (_backend.WorldSize <= 1) return [gradOutput];
        var v = gradOutput.ToVector();
        _backend.AllReduce(v, ReductionOperation.Sum);
        return [Tensor<T>.FromVector(v).Reshape(gradOutput._shape)];
    }
}

/// <summary>
/// Megatron-LM's <c>ḡ</c> conjugate operator at the OUTPUT of a tensor-parallel region
/// (Shoeybi et al. 2019, §3): all-reduce in the forward pass, identity in the backward pass.
/// </summary>
/// <remarks>
/// A row-parallel linear consumes a partitioned input and each rank produces a PARTIAL sum of the
/// full output, so the forward must all-reduce the partials into the true output <c>Y = Σ_r Y_r</c>.
/// Because every rank then receives the identical downstream gradient of that summed output, the
/// backward is the identity (the per-rank input gradients differ and are computed locally by the
/// row-parallel matmul's own tape ops).
/// </remarks>
internal sealed class ReduceFromTensorParallelRegion<T> : AutogradFunction<T>
{
    private readonly ICommunicationBackend<T> _backend;
    public ReduceFromTensorParallelRegion(ICommunicationBackend<T> backend) { _backend = backend; }

    public override Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs)
    {
        if (_backend.WorldSize <= 1) return inputs[0];
        var v = inputs[0].ToVector();
        _backend.AllReduce(v, ReductionOperation.Sum);
        return Tensor<T>.FromVector(v).Reshape(inputs[0]._shape);
    }

    public override Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput) => [gradOutput];
}
