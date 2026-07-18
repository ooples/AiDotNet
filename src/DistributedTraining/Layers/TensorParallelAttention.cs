using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Megatron-style tensor-parallel multi-head self-attention for inference/serving: the Q/K/V projections are
/// <see cref="ColumnParallelLinear{T}"/> (each rank owns a contiguous group of attention heads), each rank runs
/// scaled-dot-product attention over ONLY its local heads (no cross-rank communication inside attention), and
/// the output projection is <see cref="RowParallelLinear{T}"/> whose all-reduce sums the per-rank head
/// contributions into the full output.
/// </summary>
/// <remarks>
/// <para>
/// This is the attention counterpart to the tensor-parallel MLP pair. With head count divisible by the
/// tensor-parallel world size, the sharded block is numerically transparent: its output equals the un-sharded
/// attention's, while the Q/K/V/O weight memory and per-token compute are divided across the ranks. Exactly one
/// collective (the output projection's all-reduce) runs per attention block.
/// </para>
/// <para><b>For Beginners:</b> Attention has many "heads" that each look at the sequence a bit differently.
/// Tensor parallelism hands each GPU a subset of the heads to compute, then adds the pieces back together — so
/// a model with more heads than one GPU can handle still runs, and the answer is the same as if one GPU did it
/// all.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class TensorParallelAttention<T> : LayerBase<T>
{
    private readonly ColumnParallelLinear<T> _q;
    private readonly ColumnParallelLinear<T> _k;
    private readonly ColumnParallelLinear<T> _v;
    private readonly RowParallelLinear<T> _o;

    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _localHeads;
    private readonly bool _causal;
    private readonly double _scale;

    /// <summary>
    /// Creates a tensor-parallel attention block sharded across the ranks of <paramref name="backend"/>.
    /// </summary>
    /// <param name="backend">The tensor-parallel communication backend (defines this rank / world size).</param>
    /// <param name="embedDim">Model embedding dimension (= numHeads * headDim).</param>
    /// <param name="numHeads">Total attention heads. Must be divisible by the world size so each rank owns whole heads.</param>
    /// <param name="causal">Whether to apply a causal (autoregressive) mask.</param>
    public TensorParallelAttention(ICommunicationBackend<T> backend, int embedDim, int numHeads, bool causal = false)
        : base([embedDim], [embedDim])
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (numHeads <= 0 || embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (numHeads % backend.WorldSize != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by the tensor-parallel world size ({backend.WorldSize}) so each rank owns whole heads.",
                nameof(numHeads));

        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _localHeads = numHeads / backend.WorldSize;
        _causal = causal;
        _scale = 1.0 / Math.Sqrt(_headDim);

        // Column-parallel Q/K/V: each rank produces its head-group slice (embedDim/worldSize = localHeads*headDim),
        // NOT gathered (feeds the local attention + the row-parallel output projection's matching input shard).
        _q = new ColumnParallelLinear<T>(backend, embedDim, embedDim, gatherOutput: false);
        _k = new ColumnParallelLinear<T>(backend, embedDim, embedDim, gatherOutput: false);
        _v = new ColumnParallelLinear<T>(backend, embedDim, embedDim, gatherOutput: false);
        // Row-parallel output projection: input = this rank's head outputs; the all-reduce sums across ranks.
        _o = new RowParallelLinear<T>(backend, embedDim, embedDim);
    }

    public override bool SupportsTraining => false;

    public override long ParameterCount => _q.ParameterCount + _k.ParameterCount + _v.ParameterCount + _o.ParameterCount;

    /// <summary>Seeds all four projections from full (un-sharded) weights, slicing each rank's shard. Used to
    /// build the block from an un-sharded reference for equivalence verification.</summary>
    public void SetFromFullWeights(
        Tensor<T> qWeight, Tensor<T> qBias,
        Tensor<T> kWeight, Tensor<T> kBias,
        Tensor<T> vWeight, Tensor<T> vBias,
        Tensor<T> oWeight, Tensor<T> oBias)
    {
        _q.SetFromFullWeights(qWeight, qBias);
        _k.SetFromFullWeights(kWeight, kBias);
        _v.SetFromFullWeights(vWeight, vBias);
        _o.SetFromFullWeights(oWeight, oBias);
    }

    /// <summary>
    /// Runs tensor-parallel self-attention. Input is <c>[batch, seq, embedDim]</c>; output is the same shape
    /// (identical across ranks after the output projection's all-reduce).
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] != _embedDim)
            throw new ArgumentException($"TensorParallelAttention expects input [batch, seq, {_embedDim}]; got [{string.Join(",", input.Shape)}].", nameof(input));

        int batch = input.Shape[0];
        int seq = input.Shape[1];
        int localDim = _localHeads * _headDim;

        // Project via the column-parallel Q/K/V (2D matmul over flattened [batch*seq, embedDim]).
        var flat = input.Reshape(batch * seq, _embedDim);
        var q = _q.Forward(flat); // [batch*seq, localDim]
        var k = _k.Forward(flat);
        var v = _v.Forward(flat);

        var qs = q.AsSpan();
        var ks = k.AsSpan();
        var vs = v.AsSpan();

        // Local scaled-dot-product attention over this rank's heads, per (batch, head).
        var context = new Tensor<T>(new[] { batch * seq, localDim });
        var ctxSpan = context.AsWritableSpan();
        var scores = new double[seq];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _localHeads; h++)
            {
                int headOffset = h * _headDim;
                for (int i = 0; i < seq; i++)
                {
                    int qRow = (b * seq + i) * localDim + headOffset;
                    int limit = _causal ? i : seq - 1;

                    // scores[j] = (Q_i . K_j) * scale
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j <= limit; j++)
                    {
                        int kRow = (b * seq + j) * localDim + headOffset;
                        double dot = 0.0;
                        for (int d = 0; d < _headDim; d++)
                            dot += Convert.ToDouble(qs[qRow + d]) * Convert.ToDouble(ks[kRow + d]);
                        double s = dot * _scale;
                        scores[j] = s;
                        if (s > maxScore) maxScore = s;
                    }

                    // softmax over j in [0, limit]
                    double sumExp = 0.0;
                    for (int j = 0; j <= limit; j++)
                    {
                        double e = Math.Exp(scores[j] - maxScore);
                        scores[j] = e;
                        sumExp += e;
                    }
                    double inv = sumExp > 0 ? 1.0 / sumExp : 0.0;

                    // context_i = Σ_j softmax_j * V_j
                    int ctxRow = (b * seq + i) * localDim + headOffset;
                    for (int d = 0; d < _headDim; d++)
                    {
                        double acc = 0.0;
                        for (int j = 0; j <= limit; j++)
                        {
                            int vRow = (b * seq + j) * localDim + headOffset;
                            acc += scores[j] * inv * Convert.ToDouble(vs[vRow + d]);
                        }
                        ctxSpan[ctxRow + d] = NumOps.FromDouble(acc);
                    }
                }
            }
        }

        // Output projection: row-parallel all-reduces the per-rank head contributions into the full output.
        var outFlat = _o.Forward(context); // [batch*seq, embedDim]
        return outFlat.Reshape(batch, seq, _embedDim);
    }

    public override Vector<T> GetParameters()
    {
        var parts = new[] { _q.GetParameters(), _k.GetParameters(), _v.GetParameters(), _o.GetParameters() };
        var all = new T[ParameterCount];
        int idx = 0;
        foreach (var p in parts)
            for (int i = 0; i < p.Length; i++) all[idx++] = p[i];
        return new Vector<T>(all);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        int idx = 0;
        void Assign(LayerBase<T> layer)
        {
            long n = layer.ParameterCount;
            var slice = new T[n];
            for (int i = 0; i < n; i++) slice[i] = parameters[idx++];
            layer.SetParameters(new Vector<T>(slice));
        }
        Assign(_q); Assign(_k); Assign(_v); Assign(_o);
    }

    public override void UpdateParameters(T learningRate)
        => throw new InvalidOperationException("TensorParallelAttention is an inference-only serving block.");

    public override void ResetState() { }
}
