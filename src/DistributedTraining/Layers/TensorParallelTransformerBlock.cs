using System;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// A complete Megatron-style tensor-parallel transformer decoder block for inference/serving (pre-LayerNorm,
/// GPT-style): <c>h + Attn(LN1(h))</c> then <c>h + MLP(LN2(h))</c>, where the attention is a
/// <see cref="TensorParallelAttention{T}"/> and the MLP is the column-parallel/row-parallel Megatron pair
/// (<see cref="ColumnParallelLinear{T}"/> up + activation, <see cref="RowParallelLinear{T}"/> down). LayerNorms
/// and residuals are replicated on every rank; each attention/MLP runs exactly one all-reduce.
/// </summary>
/// <remarks>
/// <para>
/// With head count and world size compatible, the sharded block's output is numerically identical to the
/// un-sharded block — so a model too large for one GPU can be served by splitting each block across GPUs with
/// no change in results. This is the drop-in serving unit that stacks into a tensor-parallel transformer.
/// </para>
/// <para><b>For Beginners:</b> This is one full "layer" of a transformer, but with its heavy matrix-multiplies
/// split across several GPUs. Stack a few of these and you have a large language model whose weights and
/// compute are shared across your GPUs while producing exactly the same output as a single-GPU model.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
// Shape-preserving at rank 3, so the generator derives Same on every axis and no OutputAxesFor is
// written here. From ForwardTraced: the guard is "if (input.Rank != 3 || input.Shape[2] != _embedDim)
// throw", and the value returned is AddResidual(h, mlpOut) - which copies a.Shape verbatim, so a
// residual add can only ever be the shape of the tensor chained back from the input. Both sub-blocks
// are wrapped in residuals, which is what pins the whole block to its input shape.
//
// _ffnDim never appears in the contract, and that is right: the FFN widens to it inside the MLP
// sub-block and _mlpDown projects straight back to embedDim before the residual add, so the widening
// is an internal detail with no edge the caller can see. THE SHARDING IS EQUALLY INVISIBLE - each
// rank's up-projection produces only its localFfn columns and the down-projection's all-reduce sums
// them back to the full embedDim, so every rank's output is the same tensor.
//
// Rank 3 only - no BatchOptional. The guard rejects every other rank outright, so declaring the
// unbatched [Time, Features] form would claim input this layer throws on.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "Identical across ranks: each sub-block's all-reduce restores the full embedding width.")]
[AutoParameters]
public sealed partial class TensorParallelTransformerBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly LayerNormalizationLayer<T> _ln1;
    private readonly LayerNormalizationLayer<T> _ln2;
    private readonly TensorParallelAttention<T> _attention;
    private readonly ColumnParallelLinear<T> _mlpUp;
    private readonly RowParallelLinear<T> _mlpDown;

    private readonly int _embedDim;
    private readonly int _ffnDim;

    /// <summary>Creates a tensor-parallel transformer block sharded across the ranks of <paramref name="backend"/>.</summary>
    /// <param name="backend">The tensor-parallel communication backend.</param>
    /// <param name="embedDim">Model embedding dimension (= numHeads * headDim).</param>
    /// <param name="numHeads">Total attention heads (must divide the world size).</param>
    /// <param name="ffnDim">Feed-forward hidden dimension.</param>
    /// <param name="activation">FFN activation (applied in the column-parallel up-projection). Defaults to ReLU.</param>
    /// <param name="causal">Whether attention uses a causal mask.</param>
    public TensorParallelTransformerBlock(
        ICommunicationBackend<T> backend, int embedDim, int numHeads, int ffnDim,
        IActivationFunction<T>? activation = null, bool causal = false)
        : base([embedDim], [embedDim])
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));

        _embedDim = embedDim;
        _ffnDim = ffnDim;

        _ln1 = new LayerNormalizationLayer<T>(embedDim);
        _ln2 = new LayerNormalizationLayer<T>(embedDim);
        _attention = new TensorParallelAttention<T>(backend, embedDim, numHeads, causal);
        _mlpUp = new ColumnParallelLinear<T>(backend, embedDim, ffnDim, gatherOutput: false,
            activationFunction: activation ?? new AiDotNet.ActivationFunctions.ReLUActivation<T>());
        _mlpDown = new RowParallelLinear<T>(backend, ffnDim, embedDim);
    }

    public override bool SupportsTraining => false;

    /// <summary>Seeds the sharded projections (attention Q/K/V/O and MLP up/down) from full un-sharded weights.
    /// The replicated LayerNorms keep their identical (gamma=1, beta=0) initialization — set them separately via
    /// <see cref="SetParameters"/> for a trained model.</summary>
    public void SetFromFullWeights(
        Tensor<T> qW, Tensor<T> qB, Tensor<T> kW, Tensor<T> kB, Tensor<T> vW, Tensor<T> vB, Tensor<T> oW, Tensor<T> oB,
        Tensor<T> upW, Tensor<T> upB, Tensor<T> downW, Tensor<T> downB)
    {
        _attention.SetFromFullWeights(qW, qB, kW, kB, vW, vB, oW, oB);
        _mlpUp.SetFromFullWeights(upW, upB);
        _mlpDown.SetFromFullWeights(downW, downB);
    }

    /// <summary>Runs the block. Input/output are <c>[batch, seq, embedDim]</c> (output identical across ranks).</summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] != _embedDim)
            throw new ArgumentException($"TensorParallelTransformerBlock expects [batch, seq, {_embedDim}]; got [{string.Join(",", input.Shape)}].", nameof(input));

        int batch = input.Shape[0];
        int seq = input.Shape[1];

        // Attention sub-block: h = input + Attn(LN1(input)).
        var attnOut = _attention.Forward(_ln1.Forward(input));
        var h = AddResidual(input, attnOut);

        // MLP sub-block: h = h + Down(act(Up(LN2(h)))). Up/Down are 2D-matmul layers.
        var ln2 = _ln2.Forward(h);
        var flat = ln2.Reshape(batch * seq, _embedDim);
        var up = _mlpUp.Forward(flat);          // [batch*seq, localFfn] with activation
        var down = _mlpDown.Forward(up);        // [batch*seq, embedDim] all-reduced
        var mlpOut = down.Reshape(batch, seq, _embedDim);
        return AddResidual(h, mlpOut);
    }

    private Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b)
    {
        var dims = new int[a.Shape.Length];
        for (int i = 0; i < dims.Length; i++) dims[i] = a.Shape[i];
        var result = new Tensor<T>(dims);
        var ra = a.AsSpan();
        var rb = b.AsSpan();
        var rr = result.AsWritableSpan();
        for (int i = 0; i < rr.Length; i++)
            rr[i] = NumOps.Add(ra[i], rb[i]);
        return result;
    }

    public override void UpdateParameters(T learningRate)
        => throw new InvalidOperationException("TensorParallelTransformerBlock is an inference-only serving block.");

    public override void ResetState() { }
}
