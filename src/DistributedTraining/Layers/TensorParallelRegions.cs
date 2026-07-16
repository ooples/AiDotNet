using AiDotNet.Autodiff;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Constructor-initializer validation guards for the tensor-parallel layers. Called from the base()
/// initializer so a null backend or non-positive dimension is rejected BEFORE the base layer constructor
/// dereferences the backend (world size / rank) or allocates shard tensors.
/// </summary>
internal static class TensorParallelGuards
{
    /// <summary>Validates the backend and dimensions and returns <paramref name="inputSize"/> so it can be
    /// used inline in a base() initializer expression.</summary>
    public static int ValidatedInputSize<T>(ICommunicationBackend<T> backend, int inputSize, int outputSize)
    {
        if (backend is null)
            throw new System.ArgumentNullException(nameof(backend));
        if (inputSize <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(inputSize), inputSize, "Input size must be positive.");
        if (outputSize <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(outputSize), outputSize, "Output size must be positive.");
        return inputSize;
    }
}

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

/// <summary>
/// Tape-aware all-gather of a column-parallel region's output (Shoeybi et al. 2019, §3): the forward
/// AllGathers each rank's <c>[batch, localOut_r]</c> slice and re-lays it into the full
/// <c>[batch, fullOutputSize]</c> by output-column block; the backward returns THIS rank's output-column
/// slice of the incoming gradient. Implemented as an <see cref="AutogradFunction{T}"/> so the gather keeps
/// the gradient path to the local output (and hence to this rank's weight/bias shards) intact — a plain
/// value-copy gather would sever training and only forward-pass tests would notice.
/// </summary>
internal sealed class GatherFromTensorParallelRegion<T> : AutogradFunction<T>
{
    private readonly ICommunicationBackend<T> _backend;
    private readonly int _fullOutputSize;

    public GatherFromTensorParallelRegion(ICommunicationBackend<T> backend, int fullOutputSize)
    {
        _backend = backend;
        _fullOutputSize = fullOutputSize;
    }

    private int ShardCount(int rank)
        => _fullOutputSize / _backend.WorldSize + (rank < _fullOutputSize % _backend.WorldSize ? 1 : 0);

    private int ColumnStart(int rank)
    {
        int start = 0;
        for (int r = 0; r < rank; r++) start += ShardCount(r);
        return start;
    }

    public override Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs)
    {
        var local = inputs[0];                      // [batch, localOut]
        if (_backend.WorldSize <= 1) return local;
        int batch = local._shape[0];
        var gathered = _backend.AllGather(local.ToVector());
        var full = new Tensor<T>(new[] { batch, _fullOutputSize });
        int offset = 0, colBase = 0;
        for (int r = 0; r < _backend.WorldSize; r++)
        {
            int outR = ShardCount(r);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < outR; c++)
                    full[b, colBase + c] = gathered[offset + b * outR + c];
            offset += batch * outR;
            colBase += outR;
        }
        return full;
    }

    public override Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput)
    {
        // Adjoint of the column-gather: hand each rank back only its own output-column block.
        if (_backend.WorldSize <= 1) return [gradOutput];
        int batch = gradOutput._shape[0];
        int localOut = ShardCount(_backend.Rank);
        int colStart = ColumnStart(_backend.Rank);
        var slice = new Tensor<T>(new[] { batch, localOut });
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < localOut; c++)
                slice[b, c] = gradOutput[b, colStart + c];
        return [slice];
    }
}

/// <summary>
/// Identity forward, all-reduce-AVERAGE backward: synchronizes the gradient of a REPLICATED parameter
/// across the data-parallel ranks. A parameter that is stored identically on every rank (e.g. the bias of
/// an FSDP/Stage-3 sharded linear, where only the weight is reduce-scattered) would otherwise receive a
/// different local-batch gradient on each rank and drift apart. Wrapping it in this op makes every rank
/// apply the mean gradient, keeping the replica consistent — the bias analogue of the weight's
/// reduce-scatter average.
/// </summary>
internal sealed class AverageGradientAcrossRanks<T> : AutogradFunction<T>
{
    private readonly ICommunicationBackend<T> _backend;
    public AverageGradientAcrossRanks(ICommunicationBackend<T> backend) { _backend = backend; }

    public override Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs) => inputs[0];

    public override Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput)
    {
        if (_backend.WorldSize <= 1) return [gradOutput];
        var v = gradOutput.ToVector();
        _backend.AllReduce(v, ReductionOperation.Average);
        return [Tensor<T>.FromVector(v).Reshape(gradOutput._shape)];
    }
}
