using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Location-sensitive attention (Chorowski et al. 2015) as Tacotron 2 uses it (Shen et al. 2018, §2.2): additive
/// attention whose energies also read convolutional features of the previous and the cumulative attention weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <c>e_j = vᵀ tanh(W q + V h_j + U (F * [α_prev; α_cum])_j)</c>, <c>α = softmax(e)</c>, context <c>Σ_j α_j h_j</c>.
/// The paper projects inputs and location features to 128 dimensions and computes location features with 32 filters of
/// length 31; following the reference implementation (NVIDIA tacotron2 <c>Attention</c>) the location convolution and
/// every projection have no bias and the location features cover both the previous and the cumulative weights.
/// </para>
/// <para>The layer is driven step by step with <see cref="Attend"/>; <see cref="ProjectMemory"/> projects the encoder
/// output once per utterance.</para>
/// <para><b>For Beginners:</b> At every output frame the decoder decides which input characters to read next; remembering
/// where it looked before keeps it moving steadily forward through the text.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "5, 4", TestConstructorArgs = "6, 4, 3, 2, 5")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class LocationSensitiveAttentionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _queryDim;
    private readonly int _memoryDim;
    private readonly int _attentionDim;
    private readonly int _filters;
    private readonly int _kernelSize;

    private readonly BiasFreeLinearLayer<T> _query;
    private readonly BiasFreeLinearLayer<T> _memory;
    private readonly BiasFreeLinearLayer<T> _location;
    private readonly BiasFreeLinearLayer<T> _energy;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _locationKernel;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="queryDim">Width of the query (the attention RNN's output, 1024).</param>
    /// <param name="memoryDim">Width of the encoder output (512).</param>
    /// <param name="attentionDim">Width of the projections (128).</param>
    /// <param name="filters">Location convolution filters (32).</param>
    /// <param name="kernelSize">Odd location convolution length (31).</param>
    public LocationSensitiveAttentionLayer([LayerState] int queryDim, [LayerState] int memoryDim, [LayerState] int attentionDim,
        [LayerState] int filters, [LayerState] int kernelSize)
        : base(new[] { memoryDim }, new[] { memoryDim })
    {
        if (queryDim <= 0) throw new ArgumentOutOfRangeException(nameof(queryDim));
        if (memoryDim <= 0) throw new ArgumentOutOfRangeException(nameof(memoryDim));
        if (attentionDim <= 0) throw new ArgumentOutOfRangeException(nameof(attentionDim));
        if (filters <= 0) throw new ArgumentOutOfRangeException(nameof(filters));
        if (kernelSize <= 0 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd.");
        _queryDim = queryDim;
        _memoryDim = memoryDim;
        _attentionDim = attentionDim;
        _filters = filters;
        _kernelSize = kernelSize;
        _query = new BiasFreeLinearLayer<T>(queryDim, attentionDim);
        _memory = new BiasFreeLinearLayer<T>(memoryDim, attentionDim);
        _location = new BiasFreeLinearLayer<T>(filters, attentionDim);
        _energy = new BiasFreeLinearLayer<T>(attentionDim, 1);
        RegisterSubLayer(_query);
        RegisterSubLayer(_memory);
        RegisterSubLayer(_location);
        RegisterSubLayer(_energy);

        // Xavier-uniform over fan_in = 2 · k and fan_out = filters · k (NVIDIA ConvNorm, gain 1).
        double bound = Math.Sqrt(6.0 / (2 * kernelSize + filters * kernelSize));
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _locationKernel = new Tensor<T>(new[] { filters, 2, 1, kernelSize });
        for (int i = 0; i < _locationKernel.Length; i++)
            _locationKernel[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        RegisterTrainableParameter(_locationKernel, PersistentTensorRole.Weights);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The once-per-utterance memory projection <c>V h</c>, <c>[tokens, attentionDim]</c>.</summary>
    public Tensor<T> ProjectMemory(Tensor<T> memory) => _memory.Forward(memory);

    /// <summary>
    /// One attention step: the context <c>[1, memoryDim]</c> and the new weights <c>[1, tokens]</c> for the query
    /// <c>[1, queryDim]</c>, the memory <c>[tokens, memoryDim]</c>, its projection, and the previous and cumulative weights
    /// <c>[1, tokens]</c>.
    /// </summary>
    public (Tensor<T> Context, Tensor<T> Weights) Attend(Tensor<T> query, Tensor<T> memory, Tensor<T> projectedMemory,
        Tensor<T> previousWeights, Tensor<T> cumulativeWeights)
    {
        int tokens = memory.Shape[0];
        var stacked = Engine.Reshape(Engine.TensorConcatenate(new[] { previousWeights, cumulativeWeights }, 0), new[] { 1, 2, 1, tokens });
        var features = Engine.Conv2D(stacked, _locationKernel, new[] { 1, 1 }, new[] { 0, _kernelSize / 2 }, new[] { 1, 1 });
        var locationRows = Engine.TensorTranspose(Engine.Reshape(features, new[] { _filters, tokens }));          // [tokens, filters]
        var processedQuery = Engine.TensorTile(_query.Forward(query), new[] { tokens, 1 });                      // [tokens, att]
        var hidden = Engine.Tanh(Engine.TensorAdd(Engine.TensorAdd(processedQuery, projectedMemory), _location.Forward(locationRows)));
        var energies = Engine.Reshape(_energy.Forward(hidden), new[] { 1, tokens });
        var weights = Engine.TensorSoftmax(energies, axis: 1);
        return (Engine.TensorMatMul(weights, memory), weights);
    }

    /// <inheritdoc />
    /// <remarks>The first decoder step on its own: <paramref name="input"/> <c>[tokens, memoryDim]</c> is the memory,
    /// the query is zero and there are no previous weights; returns the context <c>[1, memoryDim]</c>. The decoder drives
    /// the layer step by step through <see cref="Attend"/>.</remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var memory = input.Rank == 1 ? Engine.Reshape(input, new[] { 1, input.Length }) : input;
        int tokens = memory.Shape[0];
        var zero = new Tensor<T>(new[] { 1, tokens });
        return Attend(new Tensor<T>(new[] { 1, _queryDim }), memory, ProjectMemory(memory), zero, zero).Context;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["QueryDim"] = _queryDim.ToString(inv);
        metadata["MemoryDim"] = _memoryDim.ToString(inv);
        metadata["AttentionDim"] = _attentionDim.ToString(inv);
        metadata["Filters"] = _filters.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        return metadata;
    }
}
