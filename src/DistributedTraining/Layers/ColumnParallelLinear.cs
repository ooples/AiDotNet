using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Megatron-LM column-parallel linear layer (Shoeybi et al. 2019, "Megatron-LM", §3). With the
/// framework convention <c>Y = X · Wᵀ</c> and <c>W = [outputSize, inputSize]</c>, the OUTPUT dimension
/// is partitioned across the tensor-parallel ranks: rank <c>r</c> owns the contiguous output slice
/// <c>W_r = W[start_r : start_r+localOut, :]</c> and computes its own output columns
/// <c>Y_r = X · W_rᵀ</c>. There is NO forward communication (the split output feeds a following
/// <see cref="RowParallelLinear{T}"/>); the input is wrapped in the <c>f</c> conjugate operator
/// (<see cref="CopyToTensorParallelRegion{T}"/>) whose tape backward all-reduces the input gradient
/// <c>dX = Σ_r dX_r</c>. Weight/bias-shard gradients are produced automatically by the tape from the
/// Engine matmul (no manual backward — the framework is tape-autodiff only).
/// </summary>
public sealed class ColumnParallelLinear<T> : LayerBase<T>
{
    private readonly ICommunicationBackend<T> _backend;
    private readonly CopyToTensorParallelRegion<T> _f;
    private readonly int _inputSize;
    private readonly int _fullOutputSize;
    private readonly int _localOutputSize;
    private readonly bool _gatherOutput;

    private Tensor<T> _weightShard;   // [localOutputSize, inputSize]
    private Tensor<T> _biasShard;     // [localOutputSize]

    public override bool SupportsTraining => true;
    public override long ParameterCount => _localOutputSize * (long)_inputSize + _localOutputSize;
    public int LocalOutputSize => _localOutputSize;

    public ColumnParallelLinear(
        ICommunicationBackend<T> backend,
        int inputSize,
        int outputSize,
        bool gatherOutput = false,
        IActivationFunction<T>? activationFunction = null)
        : base([inputSize],
               [gatherOutput ? outputSize : ShardCount(outputSize, backend.WorldSize, backend.Rank)],
               activationFunction ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        _backend = backend;
        _f = new CopyToTensorParallelRegion<T>(backend);
        _inputSize = inputSize;
        _fullOutputSize = outputSize;
        _localOutputSize = ShardCount(outputSize, backend.WorldSize, backend.Rank);
        _gatherOutput = gatherOutput;

        _weightShard = new Tensor<T>([_localOutputSize, inputSize]);
        _biasShard = new Tensor<T>([_localOutputSize]);
        InitializeShard();
        RegisterTrainableParameter(_weightShard, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biasShard, PersistentTensorRole.Biases);
    }

    private static int ShardCount(int total, int worldSize, int rank)
        => total / worldSize + (rank < total % worldSize ? 1 : 0);

    private int ShardStart()
        => _backend.Rank * (_fullOutputSize / _backend.WorldSize)
           + System.Math.Min(_backend.Rank, _fullOutputSize % _backend.WorldSize);

    private void InitializeShard()
    {
        // Glorot on the FULL fan-in/fan-out so the sharded init matches the unsharded model's scale.
        double scale = System.Math.Sqrt(2.0 / (_inputSize + _fullOutputSize));
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int o = 0; o < _localOutputSize; o++)
        {
            for (int i = 0; i < _inputSize; i++)
                _weightShard[o, i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
            _biasShard[o] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // f: identity forward, all-reduce backward (sums this region's input-gradient contributions).
        var x = _f.Apply(input);
        var weightT = Engine.TensorTranspose(_weightShard);                  // [inputSize, localOut]
        var linear = Engine.TensorMatMul(x, weightT);                        // [batch, localOut]
        var biased = Engine.TensorBroadcastAdd(linear, Engine.Reshape(_biasShard, new[] { 1, _localOutputSize }));
        var local = ApplyActivation(biased);
        return _gatherOutput ? GatherColumns(local, input.Shape[0]) : local;
    }

    private Tensor<T> GatherColumns(Tensor<T> local, int batch)
    {
        // AllGather returns each rank's [batch, localOut_r] flattened row-major; re-lay into
        // [batch, fullOutputSize] by output-column block. (Backward for the gather is the local slice,
        // handled when a downstream layer slices its columns; used only for standalone column-parallel.)
        var gathered = _backend.AllGather(local.ToVector());
        var full = new Tensor<T>([batch, _fullOutputSize]);
        int offset = 0, colBase = 0;
        for (int r = 0; r < _backend.WorldSize; r++)
        {
            int outR = ShardCount(_fullOutputSize, _backend.WorldSize, r);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < outR; c++)
                    full[b, colBase + c] = gathered[offset + b * outR + c];
            offset += batch * outR;
            colBase += outR;
        }
        return full;
    }

    /// <summary>Sets this rank's shard from the FULL weight/bias by slicing its output rows — used to
    /// build a parallel layer from an unsharded reference in equivalence tests.</summary>
    public void SetFromFullWeights(Tensor<T> fullWeight, Tensor<T> fullBias)
    {
        int start = ShardStart();
        for (int o = 0; o < _localOutputSize; o++)
        {
            for (int i = 0; i < _inputSize; i++)
                _weightShard[o, i] = fullWeight[start + o, i];
            _biasShard[o] = fullBias[start + o];
        }
    }

    public override Vector<T> GetParameters()
    {
        var p = new T[ParameterCount];
        int idx = 0;
        for (int o = 0; o < _localOutputSize; o++)
            for (int i = 0; i < _inputSize; i++)
                p[idx++] = _weightShard[o, i];
        for (int o = 0; o < _localOutputSize; o++)
            p[idx++] = _biasShard[o];
        return new Vector<T>(p);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int o = 0; o < _localOutputSize; o++)
            for (int i = 0; i < _inputSize; i++)
                _weightShard[o, i] = parameters[idx++];
        for (int o = 0; o < _localOutputSize; o++)
            _biasShard[o] = parameters[idx++];
    }

    // Tape-native layer: the collected trainable shards (GetTrainableParameters) are updated by the
    // optimizer's Step(TapeStepContext); the manual per-layer update is the framework's legacy path
    // and is unused here (mirrors FullyConnectedLayer, whose gradient field is likewise never set).
    public override void UpdateParameters(T learningRate)
        => throw new System.InvalidOperationException(
            "ColumnParallelLinear is tape-native: parameters are updated by the optimizer's tape Step " +
            "on the registered trainable shards, not by the legacy per-layer UpdateParameters.");

    public override void ResetState() { }
}
