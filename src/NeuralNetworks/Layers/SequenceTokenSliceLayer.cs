using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Collapses a transformer encoder's <c>[batch, seq, dim]</c> hidden
/// states down to <c>[batch, dim]</c> by selecting a single position
/// (last, first, or a fixed middle index). Used by
/// <see cref="Helpers.LayerHelper{T}.CreateDefaultTransformerLayers"/>
/// when the architecture's
/// <see cref="TransformerArchitecture{T}.SequencePooling"/> is
/// <see cref="Enums.SequencePoolingMode.LastToken"/> or
/// <see cref="Enums.SequencePoolingMode.ClsToken"/>.
/// </summary>
/// <remarks>
/// <para>
/// The reason this is a dedicated layer (instead of inlining a
/// <c>TensorSliceAxis</c> call) is so the slice participates correctly
/// in the layer chain's shape resolution + serialization passes, and
/// so it can be replaced piecewise via custom-layer overrides on
/// <see cref="TransformerArchitecture{T}.Layers"/>.
/// </para>
/// <para>
/// <b>Position semantics:</b>
/// <list type="bullet">
/// <item><c>LastToken</c> selects index <c>seq - 1</c> at runtime — the
///   actual prefix length, NOT a baked-in maxSequenceLength.</item>
/// <item><c>ClsToken</c> selects index 0 — the prepended summary token.</item>
/// </list>
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Pooling)]
[LayerTask(LayerTask.DownSampling)]
[LayerProperty(IsTrainable = false, ChangesShape = true,
    TestInputShape = "1, 4, 8",
    TestConstructorArgs = "AiDotNet.NeuralNetworks.Layers.SequenceTokenSliceLayer<double>.Position.Last")]
// Rank 3 [Batch, Time, Features] per [LayerProperty(TestInputShape = "1, 4, 8")]; the output is rank 2
// because one token is selected and the Time axis is consumed.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SequenceTokenSliceLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>
    /// Which sequence position the slice selects.
    /// </summary>
    public enum Position
    {
        /// <summary>Last position (<c>seq - 1</c>) — autoregressive LM.</summary>
        Last,
        /// <summary>First position (0) — BERT-style CLS token.</summary>
        First,
    }

    private readonly Position _position;

    /// <inheritdoc />
    /// <remarks>
    /// Selects ONE token from the sequence, so the Time axis is consumed entirely and batch and feature
    /// widths pass through untouched. <see cref="_position"/> chooses WHICH token, which changes the
    /// values but not the shape - so unlike MeanLayer's Axis, it does not enter the relation.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // Rank 3 [Batch, Time, Features] per [LayerProperty(TestInputShape = "1, 4, 8")].
        if (inputRank != 3) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
        };
    }

    /// <summary>
    /// Creates a slice layer that selects a single position from the
    /// sequence axis (axis 1 of a rank-3 input).
    /// </summary>
    public SequenceTokenSliceLayer(Position position = Position.Last)
        : base(new[] { -1, -1 }, new[] { -1 })
    {
        _position = position;
    }

    /// <inheritdoc />
    public override bool SupportsTraining => false;

    /// <inheritdoc />
    protected override void OnFirstForward(Tensor<T> input)
    {
        if (input.Shape.Length != 3)
            throw new ArgumentException(
                $"{nameof(SequenceTokenSliceLayer<T>)} requires rank-3 input " +
                $"[batch, seq, dim]; got rank {input.Shape.Length}.",
                nameof(input));

        // Layer shape metadata is per-sample and excludes the leading batch
        // dimension. Keeping the runtime batch here corrupts architecture
        // validation when the resolved layer chain is later deep-copied.
        ResolveShapes(
            new[] { input.Shape[1], input.Shape[2] },
            new[] { input.Shape[2] });
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);
        if (input.Shape.Length != 3)
            throw new ArgumentException(
                $"Forward expected rank-3 input [batch, seq, dim]; got rank {input.Shape.Length}.",
                nameof(input));

        int seq = input.Shape[1];
        if (seq <= 0)
            throw new ArgumentException(
                $"Sequence length must be positive; got {seq}.",
                nameof(input));

        int index = _position switch
        {
            Position.Last => seq - 1,
            Position.First => 0,
            _ => throw new InvalidOperationException($"Unknown Position: {_position}"),
        };

        // TensorSliceAxis is tape-tracked: the gradient w.r.t. the input
        // is zero everywhere except at the selected position, which gets
        // the full output gradient. That's the correct backward for a
        // pure index-select.
        return Engine.TensorSliceAxis(input, axis: 1, index: index);
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    /// <inheritdoc />
    public override void ResetState()
    {
        // Stateless: nothing to reset between calls.
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var m = base.GetMetadata();
        m["Position"] = _position.ToString();
        return m;
    }
}
