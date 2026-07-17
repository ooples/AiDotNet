using AiDotNet.Autodiff;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// FSDP / ZeRO Stage-3 parameter unshard operator (Rajbhandari et al. 2020; Zhao et al. 2023,
/// "PyTorch FSDP"). Forward: AllGather this rank's parameter SHARD into the full parameter, used
/// transiently for one layer's forward. Backward: ReduceScatter(Average) the full-parameter gradient
/// so each rank keeps only its shard's gradient — the reshard step. This is what makes true Stage-3
/// residency possible: only the shard is resident between steps; the full parameter exists only during
/// the layer that needs it and is released afterwards.
/// </summary>
/// <remarks>
/// The sharded parameter is the tape leaf (trainable). Applying this op yields the full parameter that
/// the layer's Engine matmul consumes, so gradients flow full-param → (ReduceScatter) → shard
/// automatically. ReduceScatter uses Average so that, under data parallelism (different data per rank),
/// each rank receives its shard of the mean gradient — the standard FSDP reduce.
/// </remarks>
internal sealed class FsdpAllGatherParameter<T> : AutogradFunction<T>
{
    private readonly ICommunicationBackend<T> _backend;
    private readonly int[] _fullShape;
    private readonly int _fullLen;
    private readonly int _shardLen;      // padded shard length (fullLen padded up to a multiple of worldSize)/worldSize

    public FsdpAllGatherParameter(ICommunicationBackend<T> backend, int[] fullShape, int shardLen)
    {
        _backend = backend;
        _fullShape = fullShape;
        _shardLen = shardLen;
        int len = 1;
        foreach (var d in fullShape) len *= d;
        _fullLen = len;
    }

    public override Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs)
    {
        // inputs[0] = this rank's shard [shardLen] (contiguous slice of the flat parameter, zero-padded).
        if (_backend.WorldSize <= 1)
            return TrimReshape(inputs[0].ToVector());

        var gathered = _backend.AllGather(inputs[0].ToVector());  // [worldSize * shardLen]
        return TrimReshape(gathered);
    }

    private Tensor<T> TrimReshape(Vector<T> flat)
    {
        // Drop the shard padding tail and reshape to the full parameter shape.
        if (flat.Length == _fullLen)
            return Tensor<T>.FromVector(flat).Reshape(_fullShape);
        var trimmed = new T[_fullLen];
        for (int i = 0; i < _fullLen; i++) trimmed[i] = flat[i];
        return Tensor<T>.FromVector(new Vector<T>(trimmed)).Reshape(_fullShape);
    }

    public override Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput)
    {
        // gradOutput = d(full parameter). Reduce-scatter (Average) so each rank keeps its shard grad.
        var gradFlat = gradOutput.ToVector();
        if (_backend.WorldSize <= 1)
            return [Tensor<T>.FromVector(gradFlat).Reshape(new[] { _shardLen })];

        // Pad to worldSize * shardLen before ReduceScatter (which requires divisibility).
        int padded = _shardLen * _backend.WorldSize;
        var ops = MathHelper.GetNumericOperations<T>();
        var buf = new T[padded];
        for (int i = 0; i < padded; i++) buf[i] = i < gradFlat.Length ? gradFlat[i] : ops.Zero;
        var myShardGrad = _backend.ReduceScatter(new Vector<T>(buf), ReductionOperation.Average);
        return [Tensor<T>.FromVector(myShardGrad).Reshape(new[] { _shardLen })];
    }
}
