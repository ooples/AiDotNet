using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// A linear layer with true ZeRO Stage-3 / FSDP parameter RESIDENCY (Rajbhandari et al. 2020; Zhao
/// et al. 2023). Each rank keeps ONLY its <c>1/worldSize</c> shard of the flat weight resident; the
/// full weight is materialized transiently by an AllGather at the start of Forward (via
/// <see cref="FsdpAllGatherParameter{T}"/>) and released as soon as the layer's matmul is done. The
/// gradient of the full weight is reduce-scattered back to the shard in the tape backward, so at rest
/// each rank stores <c>~fullParams/worldSize + outputSize</c> — peak parameter residency scales with
/// the layer, not the model. This is the residency the cache-eviction <c>CpuOffloadParams</c> path
/// only approximated.
/// </summary>
internal sealed class Stage3ShardedLinear<T> : LayerBase<T>
{
    private readonly ICommunicationBackend<T> _backend;
    private readonly FsdpAllGatherParameter<T> _unshard;
    private readonly AverageGradientAcrossRanks<T> _averageBiasGradient;
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly int _fullLen;
    private readonly int _shardLen;

    private Tensor<T> _weightShard;   // [shardLen] — the ONLY weight storage resident on this rank
    private Tensor<T> _bias;          // [outputSize] (replicated)

    public override bool SupportsTraining => true;
    public override long ParameterCount => _shardLen + _outputSize;
    public int ShardLength => _shardLen;

    public Stage3ShardedLinear(
        ICommunicationBackend<T> backend,
        int inputSize,
        int outputSize,
        IActivationFunction<T>? activationFunction = null)
        : base([TensorParallelGuards.ValidatedInputSize(backend, inputSize, outputSize)], [outputSize],
               activationFunction ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        _backend = backend;
        _inputSize = inputSize;
        _outputSize = outputSize;
        _fullLen = checked(outputSize * inputSize);   // guard against int overflow on large layers
        _shardLen = (_fullLen + backend.WorldSize - 1) / backend.WorldSize;   // ceil
        _unshard = new FsdpAllGatherParameter<T>(backend, new[] { outputSize, inputSize }, _shardLen);
        _averageBiasGradient = new AverageGradientAcrossRanks<T>(backend);

        _weightShard = new Tensor<T>([_shardLen]);
        _bias = new Tensor<T>([outputSize]);
        InitializeShard();
        RegisterTrainableParameter(_weightShard, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private void InitializeShard()
    {
        // Glorot on the FULL fan-in/fan-out; only this rank's flat slice is stored.
        double scale = System.Math.Sqrt(2.0 / (_inputSize + _outputSize));
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        int start = _backend.Rank * _shardLen;
        for (int i = 0; i < _shardLen; i++)
            _weightShard[i] = start + i < _fullLen ? NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale) : NumOps.Zero;
        for (int o = 0; o < _outputSize; o++) _bias[o] = NumOps.Zero;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Materialize the full weight just-in-time (AllGather), use it, then let it be collected.
        var fullWeight = _unshard.Apply(_weightShard);                          // [outputSize, inputSize] (transient)
        var weightT = Engine.TensorTranspose(fullWeight);                      // [inputSize, outputSize]
        var linear = Engine.TensorMatMul(input, weightT);                      // [batch, outputSize]
        // The bias is REPLICATED (not sharded like the weight, which FsdpAllGatherParameter.Backward
        // reduce-scatter-averages). Route it through the identity-forward/all-reduce-average-backward op so
        // its data-parallel gradient is averaged across ranks — otherwise each rank's local batch would
        // drift the replicated bias apart.
        var syncBias = _averageBiasGradient.Apply(_bias);
        var biased = Engine.TensorBroadcastAdd(linear, Engine.Reshape(syncBias, new[] { 1, _outputSize }));
        return ApplyActivation(biased);
    }

    /// <summary>Sets this rank's shard from the FULL weight (row-major flat slice) + the replicated bias;
    /// used to build a Stage-3 layer from an unsharded reference in equivalence tests.</summary>
    public void SetFromFullWeights(Tensor<T> fullWeight, Tensor<T> fullBias)
    {
        int start = _backend.Rank * _shardLen;
        for (int i = 0; i < _shardLen; i++)
        {
            int flat = start + i;
            _weightShard[i] = flat < _fullLen ? fullWeight[flat / _inputSize, flat % _inputSize] : NumOps.Zero;
        }
        for (int o = 0; o < _outputSize; o++) _bias[o] = fullBias[o];
    }

    public override Vector<T> GetParameters()
    {
        var p = new T[ParameterCount];
        int idx = 0;
        for (int i = 0; i < _shardLen; i++) p[idx++] = _weightShard[i];
        for (int o = 0; o < _outputSize; o++) p[idx++] = _bias[o];
        return new Vector<T>(p);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
            throw new System.ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new System.ArgumentException(
                $"Expected {ParameterCount} parameters (weight shard {_shardLen} + bias {_outputSize}), got {parameters.Length}.",
                nameof(parameters));
        int idx = 0;
        for (int i = 0; i < _shardLen; i++) _weightShard[i] = parameters[idx++];
        for (int o = 0; o < _outputSize; o++) _bias[o] = parameters[idx++];
    }

    public override void UpdateParameters(T learningRate)
        => throw new System.InvalidOperationException(
            "Stage3ShardedLinear is tape-native: the registered weight shard is updated by the " +
            "optimizer's tape Step, not by the legacy per-layer UpdateParameters.");

    public override void ResetState() { }
}
