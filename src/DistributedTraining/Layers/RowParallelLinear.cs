using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Megatron-LM row-parallel linear layer (Shoeybi et al. 2019, "Megatron-LM", §3). With the
/// framework convention <c>Y = X · Wᵀ</c> and <c>W = [outputSize, inputSize]</c>, the INPUT dimension
/// is partitioned across the tensor-parallel ranks: rank <c>r</c> owns the contiguous input slice
/// <c>W_r = W[:, start_r : start_r+localIn]</c> and consumes the matching input slice
/// <c>X_r = [batch, localIn]</c> (typically the split output of a preceding
/// <see cref="ColumnParallelLinear{T}"/>). Each rank produces a PARTIAL sum <c>Y_r = X_r · W_rᵀ</c>;
/// the <c>ḡ</c> conjugate operator (<see cref="ReduceFromTensorParallelRegion{T}"/>) all-reduces the
/// partials into the full output <c>Y = Σ_r Y_r</c> in the forward pass (identity backward). The bias
/// is replicated and added ONCE after the reduce. Weight-shard gradients come from the tape (no
/// manual backward).
/// </summary>
internal sealed class RowParallelLinear<T> : LayerBase<T>
{
    private readonly ICommunicationBackend<T> _backend;
    private readonly ReduceFromTensorParallelRegion<T> _g;
    private readonly int _fullInputSize;
    private readonly int _localInputSize;
    private readonly int _outputSize;

    private Tensor<T> _weightShard;   // [outputSize, localInputSize]
    private Tensor<T> _bias;          // [outputSize] (replicated)

    public override bool SupportsTraining => true;
    public override long ParameterCount => _outputSize * (long)_localInputSize + _outputSize;
    public int LocalInputSize => _localInputSize;

    public RowParallelLinear(
        ICommunicationBackend<T> backend,
        int inputSize,
        int outputSize,
        IActivationFunction<T>? activationFunction = null)
        : base([ShardCount(TensorParallelGuards.ValidatedInputSize(backend, inputSize, outputSize), backend.WorldSize, backend.Rank)],
               [outputSize],
               activationFunction ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        _backend = backend;
        _g = new ReduceFromTensorParallelRegion<T>(backend);
        _fullInputSize = inputSize;
        _localInputSize = ShardCount(inputSize, backend.WorldSize, backend.Rank);
        _outputSize = outputSize;

        _weightShard = new Tensor<T>([outputSize, _localInputSize]);
        _bias = new Tensor<T>([outputSize]);
        InitializeShard();
        RegisterTrainableParameter(_weightShard, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private static int ShardCount(int total, int worldSize, int rank)
        => total / worldSize + (rank < total % worldSize ? 1 : 0);

    private int ShardStart()
        => _backend.Rank * (_fullInputSize / _backend.WorldSize)
           + System.Math.Min(_backend.Rank, _fullInputSize % _backend.WorldSize);

    private void InitializeShard()
    {
        double scale = System.Math.Sqrt(2.0 / (_fullInputSize + _outputSize));
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int o = 0; o < _outputSize; o++)
        {
            for (int i = 0; i < _localInputSize; i++)
                _weightShard[o, i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
            _bias[o] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var weightT = Engine.TensorTranspose(_weightShard);                  // [localIn, outputSize]
        var partial = Engine.TensorMatMul(input, weightT);                   // [batch, outputSize] (partial)
        // ḡ: all-reduce forward (Y = Σ_r Y_r), identity backward.
        var reduced = _g.Apply(partial);
        // Bias is replicated and added ONCE after the reduce (adding before would sum it worldSize times).
        var biased = Engine.TensorBroadcastAdd(reduced, Engine.Reshape(_bias, new[] { 1, _outputSize }));
        return ApplyActivation(biased);
    }

    /// <summary>Sets this rank's shard from the FULL weight/bias by slicing its input columns; the bias
    /// is replicated. Used to build a parallel layer from an unsharded reference in equivalence tests.</summary>
    public void SetFromFullWeights(Tensor<T> fullWeight, Tensor<T> fullBias)
    {
        int start = ShardStart();
        for (int o = 0; o < _outputSize; o++)
        {
            for (int i = 0; i < _localInputSize; i++)
                _weightShard[o, i] = fullWeight[o, start + i];
            _bias[o] = fullBias[o];
        }
    }

    public override Vector<T> GetParameters()
    {
        var p = new T[ParameterCount];
        int idx = 0;
        for (int o = 0; o < _outputSize; o++)
            for (int i = 0; i < _localInputSize; i++)
                p[idx++] = _weightShard[o, i];
        for (int o = 0; o < _outputSize; o++)
            p[idx++] = _bias[o];
        return new Vector<T>(p);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
            throw new System.ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new System.ArgumentException(
                $"Expected {ParameterCount} parameters (out {_outputSize} x localIn {_localInputSize} + bias {_outputSize}), got {parameters.Length}.",
                nameof(parameters));
        int idx = 0;
        for (int o = 0; o < _outputSize; o++)
            for (int i = 0; i < _localInputSize; i++)
                _weightShard[o, i] = parameters[idx++];
        for (int o = 0; o < _outputSize; o++)
            _bias[o] = parameters[idx++];
    }

    public override void UpdateParameters(T learningRate)
        => throw new System.InvalidOperationException(
            "RowParallelLinear is tape-native: parameters are updated by the optimizer's tape Step " +
            "on the registered trainable shards, not by the legacy per-layer UpdateParameters.");

    public override void ResetState() { }
}
